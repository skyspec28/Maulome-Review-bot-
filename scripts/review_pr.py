#!/usr/bin/env python3
"""
Maulome PR Reviewer — powered by Google Gemini.

Workflow:
1. Fetch PR files/diff via GitHub API.
2. Build diff context with line-numbered file content so Gemini can cite exact lines.
3. Ask Gemini for structured JSON findings (path, line, category, confidence, description, suggestion).
4. Validate each finding's line number against the actual diff hunks (GitHub only accepts
   inline comments on lines that are inside a diff hunk).
5. Post a GitHub PR review (POST /pulls/{pr}/reviews) with:
   - Inline comments anchored to specific changed lines.
   - Where Gemini provides a replacement, a ```suggestion``` block is included —
     GitHub renders this as a native "Commit suggestion" button.
   - Findings that can't be anchored to a diff line fall back to the top-level summary.
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


def dismiss_existing_summary(repo, pr_number):
    """Delete any previous Maulome top-level summary comment so we stay tidy."""
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
    Submit a GitHub PR review.
    inline_comments: list of dicts {path, line, side, body}.
    Falls back to summary-only if the inline payload is rejected (bad line numbers).
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
        print(f"Review POST failed ({resp.status_code}): {resp.text[:400]}")
        print("Retrying without inline comments…")
        payload["comments"] = []
        resp = requests.post(url, headers=gh_headers(), json=payload)
        resp.raise_for_status()
        print("Posted summary-only review (inline comments dropped).")
    else:
        print(f"Posted PR review with {len(inline_comments)} inline comment(s).")
    return resp.json()


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------
def should_skip(filename):
    return any(re.search(pat, filename) for pat in SKIP_PATTERNS)


def parse_hunk_lines(patch: str) -> set:
    """
    Return the set of RIGHT-SIDE (new-file) line numbers present in a unified diff patch.
    GitHub only accepts inline review comments on lines that appear in a diff hunk.
    """
    valid: set = set()
    if not patch:
        return valid
    current = 0
    in_hunk = False
    for raw in patch.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            current = int(m.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw.startswith("-"):
            continue           # deletion — doesn't advance new-file line number
        valid.add(current)
        current += 1           # addition (+) or context line both advance it
    return valid


def snap_to_nearest_hunk_line(line: int, valid_lines: set, window: int = 5) -> int:
    """
    If `line` is not in `valid_lines`, find the closest valid line within ±window.
    Returns the snapped line number, or 0 if nothing is close enough.
    This prevents valid findings from being silently demoted to the summary
    because Gemini's line number is off by 1-2 due to diff context.
    """
    if line in valid_lines:
        return line
    for delta in range(1, window + 1):
        if line + delta in valid_lines:
            return line + delta
        if line - delta in valid_lines:
            return line - delta
    return 0


def build_diff_context(repo, files, head_sha):
    """
    Returns (diff_context_str, patch_lines_dict).

    diff_context_str: sent to Gemini. Each file gets:
      - The raw unified diff patch (shows what changed).
      - The full post-merge file content with 'NNNNN | code' line numbers
        so Gemini can cite exact line numbers reliably.

    patch_lines_dict: filename → set[int] of valid diff line numbers.
    """
    parts = []
    total = 0
    patch_lines = {}   # filename → set of valid line numbers

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
                truncated = raw[:MAX_FILE_CHARS]
                # Number every line so Gemini can cite exact line numbers
                numbered = "\n".join(
                    f"{i + 1:>5} | {line}"
                    for i, line in enumerate(truncated.splitlines())
                )
                content = numbered
                if len(raw) > MAX_FILE_CHARS:
                    content += f"\n... [truncated — {len(raw) - MAX_FILE_CHARS} chars omitted]"

        chunk = f"### `{filename}` ({status})\n"
        if patch:
            chunk += f"**Diff:**\n```diff\n{patch}\n```\n"
        if content:
            chunk += f"**Full file (line-numbered):**\n```\n{content}\n```\n"

        if total + len(chunk) > MAX_TOTAL_CHARS:
            parts.append(f"### `{filename}` — skipped (total size limit reached)\n")
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts), patch_lines


# ---------------------------------------------------------------------------
# Gemini API — returns list[dict] findings
# ---------------------------------------------------------------------------
def call_gemini(pr_title, pr_body, diff_context) -> list:
    system_prompt = (
        "You are an expert code reviewer. You will be given a pull request title, "
        "description, and changed files. Each file's full content is shown with line numbers "
        "in the format 'NNNNN | code' — use these EXACT numbers when citing lines. "
        "Unified diffs are also shown so you know what actually changed.\n\n"
        "Analyze every changed file for:\n"
        "1. Correctness — logical errors, missing error handling, edge cases, state inconsistency, race conditions.\n"
        "2. Security — credential leaks, injection risks, unsafe deserialization, improper validation.\n"
        "3. Performance — inefficient queries (N+1), memory leaks, slow algorithms, missing indexes.\n"
        "4. Breaking Changes — API signature changes, DB schema changes, removed functionality.\n\n"
        "OUTPUT RULES — follow ALL of these strictly:\n"
        "- Return ONLY a raw JSON array. No markdown fences, no prose outside the array.\n"
        "- Each element is a JSON object with EXACTLY these keys (no extras, no omissions):\n"
        '  {"path": str, "line": int, "category": str, "confidence": int, "description": str, "suggestion": str}\n'
        "- `path`: the exact file path as shown in the diff header (e.g. 'backend/scripts/foo.py').\n"
        "- `line`: a line number from the NUMBERED FILE CONTENT (e.g. from '  142 | some_code()'). "
        "It MUST be a line that appears in the unified diff patch for that file (a + line or context line). "
        "Do NOT use line 0 — always pick the most specific relevant line inside the diff.\n"
        "- `category`: one of 'Correctness', 'Security', 'Performance', 'Breaking Change'.\n"
        "- `confidence`: integer 0–100. ONLY include findings with confidence >= 80.\n"
        "- `description`: 2–4 sentence explanation of the problem. Do NOT include code here.\n"
        "- `suggestion`: REQUIRED for every finding. This is a direct drop-in replacement for "
        "the code at `line` — it will be rendered as a GitHub 'Apply suggestion' button. "
        "Preserve indentation exactly. If the fix requires removing the line, use an empty string. "
        "If a fix genuinely cannot be expressed as a single-line replacement, provide the closest "
        "corrected version of that line.\n"
        "- Never set `suggestion` to null, a prose string, or an explanation — ALWAYS valid code.\n"
        "- Only report issues in lines that are ADDED or CHANGED in the diff (+ lines), not pre-existing context.\n"
        "- No style preferences, no formatting nitpicks, no naming suggestions.\n"
        "- Return [] if there are zero high-confidence findings.\n"
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
        data = resp.json()
        candidate = data["candidates"][0]

        finish_reason = candidate.get("finishReason", "")
        if finish_reason not in ("STOP", ""):
            print(f"Warning: Gemini finishReason={finish_reason!r} — response may be truncated.")

        raw_text = candidate["content"]["parts"][0]["text"].strip()
        # Strip any accidental markdown fences
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            findings = json.loads(raw_text)
            if not isinstance(findings, list):
                raise ValueError("Expected a JSON array.")
            return findings
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse Gemini response as JSON: {e}")
            print(f"Raw (first 500 chars):\n{raw_text[:500]}")
            # Surface the raw response as a fallback finding
            return [{
                "path": "", "line": 0, "category": "Correctness", "confidence": 80,
                "description": (
                    "The review model returned an unparseable response. "
                    "Raw output:\n\n```\n" + raw_text[:2000] + "\n```"
                ),
                "suggestion": None,
            }]

    sys.exit(f"Gemini API still returning 429 after {MAX_RETRIES} retries.")


# ---------------------------------------------------------------------------
# Build inline comments + general findings
# ---------------------------------------------------------------------------
def build_review_payload(findings: list, patch_lines: dict):
    """
    Splits findings into:
    - inline_comments: anchored to a valid diff line, each with a ```suggestion``` block.
    - general_findings: findings that cannot be anchored — go in the top-level summary body.

    Uses snap_to_nearest_hunk_line() to tolerate ±5 line off-by-one errors from Gemini
    instead of silently demoting the finding to the summary.
    """
    inline_comments = []
    general_findings = []

    for f in findings:
        path        = f.get("path", "")
        line        = f.get("line", 0)
        category    = f.get("category", "General")
        confidence  = f.get("confidence", 80)
        description = f.get("description", "")
        suggestion  = f.get("suggestion") or ""

        # Snap to the nearest valid hunk line (tolerates ±5 off-by-one from Gemini)
        valid_lines = patch_lines.get(path, set())
        anchored_line = snap_to_nearest_hunk_line(line, valid_lines) if path and line > 0 else 0

        body = f"**[{category}]** (Confidence: {confidence}%)\n\n{description}"
        if suggestion.strip():
            body += f"\n\n```suggestion\n{suggestion}\n```"

        if path and anchored_line > 0:
            inline_comments.append({
                "path": path,
                "line": anchored_line,
                "side": "RIGHT",
                "body": body,
            })
        else:
            # Genuine whole-file finding or completely outside all hunks
            location = f"`{path}` line {line}" if path else "General"
            entry = f"### [{category}] — {location} (Confidence: {confidence}%)\n\n{description}"
            if suggestion.strip():
                entry += f"\n\n**Suggested fix:**\n```\n{suggestion}\n```"
            general_findings.append(entry)

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

    # Belt-and-braces confidence filter
    findings = [f for f in findings if f.get("confidence", 0) >= 80]
    print(f"After confidence filter: {len(findings)} finding(s).")

    inline_comments, general_findings = build_review_payload(findings, patch_lines)
    print(f"Inline: {len(inline_comments)}, General fallback: {len(general_findings)}")

    # Build top-level review body
    if not findings:
        verdict = "✅ **No high-confidence issues found — looks good to merge.**"
    else:
        verdict = "⚠️ **Review complete — see inline comments below.**"

    summary_parts = [
        COMMENT_MARKER,
        "## 🤖 Maulome PR Review",
        f"*Model: `{GEMINI_MODEL}`*\n",
        verdict,
    ]

    if general_findings:
        summary_parts.append("\n---\n### 📋 General Findings (not anchored to a specific diff line)\n")
        summary_parts.extend(general_findings)

    summary_parts.append(
        "\n---\n*Powered by [Maulome Review Bot](https://github.com/skyspec28/Maulome-Review-bot-)*"
    )

    summary_body = "\n".join(summary_parts)

    # Remove stale summary comment from previous runs
    dismiss_existing_summary(repo, pr_number)

    post_pr_review(repo, pr_number, head_sha, summary_body, inline_comments)


if __name__ == "__main__":
    main()
