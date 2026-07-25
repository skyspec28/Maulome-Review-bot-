#!/usr/bin/env python3
"""
Maulome PR Reviewer — powered by Google Gemini.

Workflow:
1. Fetch PR files/diff via GitHub API.
2. Send diff + context to Gemini, asking for structured JSON findings.
3. Validate each finding's line number against the actual diff hunks.
4. Post an inline GitHub PR review with:
   - Inline comments anchored to changed lines (with ```suggestion``` blocks
     where Gemini supplies a replacement — renders a "Commit suggestion" button).
   - A top-level summary comment for general findings that aren't line-anchored.
"""

import os
import re
import sys
import json
import time
import requests

# ---------------------------------------------------------------------------
# Configuration (override via env vars in the workflow)
# ---------------------------------------------------------------------------
GITHUB_TOKEN    = os.environ["GITHUB_TOKEN"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
MAX_FILE_CHARS  = int(os.environ.get("MAX_FILE_CHARS",  "8000"))
MAX_TOTAL_CHARS = int(os.environ.get("MAX_TOTAL_CHARS", "60000"))
MAX_RETRIES     = int(os.environ.get("MAX_RETRIES", "5"))

GITHUB_API      = "https://api.github.com"
GEMINI_API_URL  = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
COMMENT_MARKER  = "<!-- maulome-pr-reviewer -->"

# Files to skip
SKIP_PATTERNS = [
    r"\.lock$",
    r"package-lock\.json$",
    r"yarn\.lock$",
    r"pnpm-lock\.yaml$",
    r"\.min\.(js|css)$",
    r"dist/",
    r"build/",
    r"__pycache__/",
    r"\.pyc$",
    r"\.png$", r"\.jpg$", r"\.jpeg$", r"\.gif$", r"\.svg$", r"\.ico$",
    r"\.pdf$", r"\.zip$", r"\.tar$",
]


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------
def gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_pr_info():
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path or not os.path.exists(event_path):
        sys.exit("GITHUB_EVENT_PATH not set or file missing.")
    with open(event_path) as f:
        event = json.load(f)
    pr = event.get("pull_request") or event.get("issue")
    if not pr:
        sys.exit("Could not find pull_request in event payload.")
    repo_full = event["repository"]["full_name"]
    pr_number = pr["number"]
    base_sha  = pr["base"]["sha"]
    head_sha  = pr["head"]["sha"]
    pr_title  = pr.get("title", "")
    pr_body   = pr.get("body") or ""
    return repo_full, pr_number, base_sha, head_sha, pr_title, pr_body


def get_pr_files(repo, pr_number):
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files"
    resp = requests.get(url, headers=gh_headers())
    resp.raise_for_status()
    return resp.json()


def get_file_content(repo, path, ref):
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}?ref={ref}"
    resp = requests.get(url, headers=gh_headers())
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("encoding") == "base64":
        import base64
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return None


def dismiss_existing_reviews(repo, pr_number):
    """Find any previous Maulome review and delete its top-level summary comment."""
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    resp = requests.get(url, headers=gh_headers())
    resp.raise_for_status()
    for c in resp.json():
        if COMMENT_MARKER in c.get("body", ""):
            del_url = f"{GITHUB_API}/repos/{repo}/issues/comments/{c['id']}"
            requests.delete(del_url, headers=gh_headers())
            print(f"Deleted stale summary comment (id={c['id']}).")


def post_pr_review(repo, pr_number, head_sha, summary_body, inline_comments):
    """
    Post a GitHub PR review with optional inline comments.
    inline_comments: list of dicts with keys: path, line, body (may include ```suggestion``` block).
    """
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews"
    payload = {
        "commit_id": head_sha,
        "event": "COMMENT",
        "body": summary_body,
        "comments": inline_comments,
    }
    resp = requests.post(url, headers=gh_headers(), json=payload)
    if not resp.ok:
        # If inline comments fail (e.g. bad line numbers slipped through),
        # fall back to submitting without them so we don't lose the summary.
        print(f"Review POST failed ({resp.status_code}): {resp.text[:300]}")
        print("Retrying without inline comments…")
        payload["comments"] = []
        resp = requests.post(url, headers=gh_headers(), json=payload)
        resp.raise_for_status()
        print("Posted summary-only review (inline comments were dropped).")
    else:
        print(f"Posted PR review with {len(inline_comments)} inline comment(s).")
    return resp.json()


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------
def should_skip(filename):
    return any(re.search(pat, filename) for pat in SKIP_PATTERNS)


def parse_hunk_lines(patch: str) -> set[int]:
    """
    Parse a unified diff patch string and return the set of RIGHT-SIDE (new-file)
    line numbers that appear in the diff hunks. Only these lines are valid targets
    for inline comments — GitHub rejects comments on lines outside hunk context.
    """
    valid_lines: set[int] = set()
    if not patch:
        return valid_lines

    current_line = 0
    for raw_line in patch.splitlines():
        # Hunk header: @@ -old_start,old_count +new_start,new_count @@
        hunk_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw_line)
        if hunk_match:
            current_line = int(hunk_match.group(1))
            continue
        if raw_line.startswith("-"):
            # Deletion — doesn't advance new-file line number
            continue
        if raw_line.startswith("+"):
            valid_lines.add(current_line)
            current_line += 1
        else:
            # Context line — advances both old and new
            valid_lines.add(current_line)
            current_line += 1

    return valid_lines


def build_diff_context(repo, files, head_sha):
    """
    Build a rich text context string for Gemini, AND return a mapping of
    filename → set of valid diff line numbers for later validation.
    """
    parts = []
    total = 0
    patch_lines: dict[str, set[int]] = {}   # filename → valid line numbers

    for f in files:
        filename = f["filename"]
        status   = f["status"]
        if should_skip(filename):
            continue

        patch = f.get("patch", "")
        patch_lines[filename] = parse_hunk_lines(patch)

        content = ""
        if status != "removed":
            raw = get_file_content(repo, filename, head_sha)
            if raw:
                content = raw[:MAX_FILE_CHARS]
                if len(raw) > MAX_FILE_CHARS:
                    content += f"\n... [truncated — {len(raw) - MAX_FILE_CHARS} chars omitted]"

        chunk = f"### `{filename}` ({status})\n"
        if patch:
            chunk += f"**Diff:**\n```diff\n{patch}\n```\n"
        if content:
            ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
            chunk += f"**Full file (line numbers shown for reference):**\n```{ext}\n"
            for i, line in enumerate(content.splitlines(), start=1):
                chunk += f"{i}: {line}\n"
            chunk += "```\n"

        if total + len(chunk) > MAX_TOTAL_CHARS:
            parts.append(f"### `{filename}` — skipped (total size limit reached)\n")
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts), patch_lines


# ---------------------------------------------------------------------------
# Gemini API call — returns structured JSON findings
# ---------------------------------------------------------------------------
def call_gemini(pr_title, pr_body, diff_context) -> list[dict]:
    system_prompt = (
        "You are an expert code reviewer. You will be given a pull request title, "
        "description, and changed files showing line-numbered source code and unified diffs.\n\n"
        "Your job is to produce a thorough code review as a JSON array. "
        "Analyze every file in the diff for:\n"
        "1. Correctness — logical errors, missing error handling, edge cases, state inconsistency, race conditions.\n"
        "2. Security — credential leaks, injection risks, unsafe deserialization, improper validation.\n"
        "3. Performance — inefficient queries (N+1), memory leaks, slow algorithms, missing indexes.\n"
        "4. Breaking Changes — API signature changes, DB schema changes, removed functionality.\n\n"
        "OUTPUT RULES (strictly follow):\n"
        "- Return ONLY a raw JSON array. No markdown fences, no prose, no explanation outside the array.\n"
        "- Each element must be an object with these exact keys:\n"
        '  {"path": str, "line": int, "category": str, "confidence": int, "description": str, "suggestion": str|null}\n'
        "- `path`: the file path exactly as shown in the diff headers.\n"
        "- `line`: the RIGHT-SIDE (new file) line number from the diff. Must be a line that appears "
        "in a +/context line of the diff for that file. Use 0 if the finding applies to the whole file.\n"
        "- `category`: one of 'Correctness', 'Security', 'Performance', 'Breaking Change'.\n"
        "- `confidence`: integer 0-100. Only include findings with confidence >= 80.\n"
        "- `description`: clear explanation of the issue (2-5 sentences max).\n"
        "- `suggestion`: the exact replacement code for that line (or null if not applicable). "
        "This will be rendered as a GitHub suggestion block — it must be a drop-in line replacement, not prose.\n"
        "- Omit any finding with confidence < 80.\n"
        "- Do NOT include style preferences, formatting nitpicks, or minor naming suggestions.\n"
        "- If there are no high-confidence findings, return an empty array: []\n"
    )

    user_content = (
        f"## PR Title\n{pr_title}\n\n"
        f"## PR Description\n{pr_body or '_No description provided._'}\n\n"
        f"## Changed Files (Diff & Line-Numbered Source)\n{diff_context}"
    )

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192,
        },
    }

    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.post(url, json=payload, timeout=180)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 0))
            wait = retry_after if retry_after > 0 else min(20 * (2 ** (attempt - 1)), 120)
            print(f"Rate limited (429). Attempt {attempt}/{MAX_RETRIES}. Waiting {wait}s…")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        # Strip any accidental markdown fences the model adds around the JSON
        raw_text = raw_text.strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            findings = json.loads(raw_text)
            if not isinstance(findings, list):
                raise ValueError("Gemini returned a non-list JSON value.")
            return findings
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse Gemini response as JSON: {e}")
            print(f"Raw response (first 500 chars):\n{raw_text[:500]}")
            # Return a single general finding so we at least surface the raw output
            return [{
                "path": "",
                "line": 0,
                "category": "Correctness",
                "confidence": 80,
                "description": (
                    "The review model returned an unparseable response. "
                    "Raw output:\n\n```\n" + raw_text[:2000] + "\n```"
                ),
                "suggestion": None,
            }]

    sys.exit(f"Gemini API still returning 429 after {MAX_RETRIES} retries.")


# ---------------------------------------------------------------------------
# Build the review payload
# ---------------------------------------------------------------------------
def build_review_payload(findings: list[dict], patch_lines: dict[str, set[int]]):
    """
    Split findings into:
    - inline_comments: anchored to a specific valid diff line (with optional suggestion block)
    - general_findings: whole-file or unanchorable findings → go in the top-level summary
    """
    inline_comments = []
    general_findings = []

    for f in findings:
        path = f.get("path", "")
        line = f.get("line", 0)
        category   = f.get("category", "General")
        confidence = f.get("confidence", 80)
        description = f.get("description", "")
        suggestion  = f.get("suggestion")

        # Build the comment body
        body = f"**[{category}]** (Confidence: {confidence}%)\n\n{description}"
        if suggestion:
            body += f"\n\n```suggestion\n{suggestion}\n```"

        # Validate: line must be in the actual diff hunk for this file
        valid_lines = patch_lines.get(path, set())
        if path and line > 0 and line in valid_lines:
            inline_comments.append({
                "path": path,
                "line": line,
                "side": "RIGHT",
                "body": body,
            })
        else:
            # Fall back to a general finding (will appear in the top-level summary)
            location = f"`{path}` line {line}" if path else "General"
            general_findings.append(f"### [{category}] — {location} (Confidence: {confidence}%)\n\n{description}")
            if suggestion:
                general_findings[-1] += f"\n\n**Suggested fix:**\n```\n{suggestion}\n```"

    return inline_comments, general_findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    repo, pr_number, base_sha, head_sha, pr_title, pr_body = get_pr_info()
    print(f"Reviewing PR #{pr_number} in {repo}  ({base_sha[:7]}..{head_sha[:7]})")

    files = get_pr_files(repo, pr_number)
    print(f"Files changed: {len(files)}")

    diff_context, patch_lines = build_diff_context(repo, files, head_sha)
    if not diff_context.strip():
        print("No reviewable files found — skipping.")
        return

    print(f"Calling {GEMINI_MODEL} via Google AI …")
    findings = call_gemini(pr_title, pr_body, diff_context)
    print(f"Gemini returned {len(findings)} finding(s).")

    # Filter to >= 80% confidence (model should already do this, belt + braces)
    findings = [f for f in findings if f.get("confidence", 0) >= 80]
    print(f"After confidence filter: {len(findings)} finding(s).")

    inline_comments, general_findings = build_review_payload(findings, patch_lines)
    print(f"Inline: {len(inline_comments)}, General: {len(general_findings)}")

    # Build the top-level review body
    verdict_line = ""
    if not findings:
        verdict_line = "✅ **No high-confidence issues found — looks good to merge.**"
    else:
        verdict_line = "⚠️ **Review complete — see inline comments and findings below.**"

    summary_parts = [
        COMMENT_MARKER,
        f"## 🤖 Maulome PR Review",
        f"*Model: `{GEMINI_MODEL}`*\n",
        verdict_line,
    ]

    if general_findings:
        summary_parts.append("\n---\n### 📋 General Findings (not anchored to a specific line)\n")
        summary_parts.extend(general_findings)

    summary_parts.append(
        "\n---\n*Powered by [Maulome Review Bot](https://github.com/skyspec28/Maulome-Review-bot-)*"
    )

    summary_body = "\n".join(summary_parts)

    # Delete old summary comment from previous runs (inline review comments
    # are immutable per-review, so we just post a new review each time)
    dismiss_existing_reviews(repo, pr_number)

    post_pr_review(repo, pr_number, head_sha, summary_body, inline_comments)


if __name__ == "__main__":
    main()
