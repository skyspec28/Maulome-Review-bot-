#!/usr/bin/env python3
"""
Maulome PR Reviewer — powered by Google Gemini.

Workflow:
1. Fetch PR files and diffs via GitHub API.
2. Parse each diff patch to extract the exact set of valid RIGHT-SIDE line numbers
   (lines that appear in a diff hunk — the only lines GitHub accepts inline comments on).
3. Send Gemini: the diff, the line-numbered full file, AND a pre-computed list of
   "valid anchor lines" per file — so it only ever cites lines GitHub will accept.
4. Gemini returns structured JSON findings.
5. Post a GitHub PR Review (POST /pulls/{pr}/reviews) with:
   - Inline comments anchored to changed lines, each with a ```suggestion``` block
     that GitHub renders as a one-click "Apply suggestion" button.
   - Findings that can't be anchored go in the top-level review body as a fallback.
"""

import os
import re
import sys
import json
import time
import requests

# ---------------------------------------------------------------------------
# Configuration
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

SKIP_PATTERNS = [
    r"\.lock$", r"package-lock\.json$", r"yarn\.lock$", r"pnpm-lock\.yaml$",
    r"\.min\.(js|css)$", r"dist/", r"build/", r"__pycache__/", r"\.pyc$",
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
    return (
        event["repository"]["full_name"],
        pr["number"],
        pr["base"]["sha"],
        pr["head"]["sha"],
        pr.get("title", ""),
        pr.get("body") or "",
    )


def get_pr_files(repo, pr_number):
    resp = requests.get(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files",
        headers=gh_headers(),
    )
    resp.raise_for_status()
    return resp.json()


def get_file_content(repo, path, ref):
    resp = requests.get(
        f"{GITHUB_API}/repos/{repo}/contents/{path}?ref={ref}",
        headers=gh_headers(),
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("encoding") == "base64":
        import base64
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return None


def dismiss_existing_summary(repo, pr_number):
    resp = requests.get(
        f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
        headers=gh_headers(),
    )
    resp.raise_for_status()
    for c in resp.json():
        if COMMENT_MARKER in c.get("body", ""):
            requests.delete(
                f"{GITHUB_API}/repos/{repo}/issues/comments/{c['id']}",
                headers=gh_headers(),
            )
            print(f"Deleted stale summary comment (id={c['id']}).")


def post_pr_review(repo, pr_number, head_sha, summary_body, inline_comments):
    """
    Post the review. Falls back to summary-only if inline comments are rejected
    (e.g. stale head_sha after a force-push).
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


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------
def should_skip(filename):
    return any(re.search(pat, filename) for pat in SKIP_PATTERNS)


def parse_hunk_lines(patch: str) -> set:
    """
    Returns the set of RIGHT-SIDE (new-file) line numbers that appear anywhere
    in the diff patch (both added lines and context lines).
    These are the ONLY lines GitHub accepts inline review comments on.
    """
    valid: set = set()
    if not patch:
        return valid
    in_hunk = False
    current = 0
    for raw in patch.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            current = int(m.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw.startswith("-"):
            continue  # deletion — doesn't move new-file line counter
        valid.add(current)
        current += 1
    return valid


def snap_to_valid(line: int, valid: set, window: int = 10) -> int:
    """
    If `line` is not in `valid`, find the nearest valid line within ±window.
    Returns 0 if nothing is close enough (finding becomes a general comment).
    """
    if line in valid:
        return line
    for d in range(1, window + 1):
        if (line + d) in valid:
            return line + d
        if (line - d) in valid:
            return line - d
    return 0


# ---------------------------------------------------------------------------
# Build diff context for Gemini
# ---------------------------------------------------------------------------
def build_diff_context(repo, files, head_sha):
    """
    Returns (context_str, patch_lines_map).

    context_str is sent to Gemini. For each file it contains:
      - The raw unified diff patch (what actually changed).
      - The full file with line numbers in 'NNNNN | code' format.
      - A "VALID ANCHOR LINES" list — the exact line numbers Gemini MUST choose from
        when citing a line. This is the key addition that eliminates guessing.

    patch_lines_map: {filename: set of valid line numbers}
    """
    parts = []
    total = 0
    patch_lines_map = {}

    for f in files:
        filename = f["filename"]
        status   = f["status"]
        if should_skip(filename):
            continue

        patch = f.get("patch", "")
        valid = parse_hunk_lines(patch)
        patch_lines_map[filename] = valid

        content = ""
        if status != "removed":
            raw = get_file_content(repo, filename, head_sha)
            if raw:
                truncated = raw[:MAX_FILE_CHARS]
                content = "\n".join(
                    f"{i + 1:>5} | {line}"
                    for i, line in enumerate(truncated.splitlines())
                )
                if len(raw) > MAX_FILE_CHARS:
                    content += f"\n... [truncated — {len(raw) - MAX_FILE_CHARS} chars omitted]"

        chunk = f"### `{filename}` ({status})\n"
        if patch:
            chunk += f"**Diff:**\n```diff\n{patch}\n```\n"
        if content:
            chunk += f"**Full file (line-numbered):**\n```\n{content}\n```\n"
        if valid:
            # Tell Gemini explicitly which lines it can anchor to.
            sorted_valid = sorted(valid)
            chunk += f"**VALID ANCHOR LINES (you MUST use one of these as `line`):** {sorted_valid}\n\n"

        if total + len(chunk) > MAX_TOTAL_CHARS:
            parts.append(f"### `{filename}` — skipped (total size limit reached)\n")
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts), patch_lines_map


# ---------------------------------------------------------------------------
# Gemini call — structured JSON findings
# ---------------------------------------------------------------------------
def call_gemini(pr_title, pr_body, diff_context) -> list:
    system_prompt = (
        "You are an expert code reviewer. You are given:\n"
        "- A unified diff showing what changed (the `+` lines are additions).\n"
        "- The full file with line numbers in `NNNNN | code` format.\n"
        "- A **VALID ANCHOR LINES** list per file — these are the ONLY line numbers "
        "GitHub will accept for inline comments. You MUST pick `line` from this list.\n\n"
        "Review every changed file for:\n"
        "1. Correctness — logic errors, missing error handling, edge cases, state inconsistency, race conditions.\n"
        "2. Security — credential leaks, injection risks, unsafe deserialization, improper validation.\n"
        "3. Performance — N+1 queries, memory leaks, slow algorithms, missing indexes.\n"
        "4. Breaking Changes — API signature changes, DB schema changes, removed functionality.\n\n"
        "OUTPUT — return ONLY a raw JSON array, no markdown fences, no prose outside the array:\n"
        "[\n"
        "  {\n"
        '    "path": "exact/file/path.py",\n'
        '    "line": <integer from VALID ANCHOR LINES for this file>,\n'
        '    "category": "Correctness" | "Security" | "Performance" | "Breaking Change",\n'
        '    "confidence": <integer 80-100>,\n'
        '    "description": "<2-4 sentences explaining the problem — NO code here>",\n'
        '    "suggestion": "<exact drop-in replacement for that line — preserving indentation — '
        'will appear as a GitHub Apply-suggestion button>"\n'
        "  }\n"
        "]\n\n"
        "RULES:\n"
        "- `line` MUST be a number from the VALID ANCHOR LINES list for that file. No exceptions.\n"
        "- `suggestion` is required for every finding. It must be valid code matching the language "
        "of the file — not prose, not an explanation. It replaces the exact line cited.\n"
        "- Only report code that was ADDED or CHANGED in the diff (+ lines). Ignore unchanged context.\n"
        "- Only include findings with confidence >= 80.\n"
        "- No style nitpicks, no formatting preferences, no naming suggestions.\n"
        "- Return [] if there are no high-confidence findings.\n"
    )

    user_content = (
        f"## PR Title\n{pr_title}\n\n"
        f"## PR Description\n{pr_body or '_No description provided._'}\n\n"
        f"## Changed Files\n{diff_context}"
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

        if resp.status_code in (500, 502, 503, 504):
            wait = min(15 * (2 ** (attempt - 1)), 120)
            print(f"Gemini {resp.status_code} (transient). Attempt {attempt}/{MAX_RETRIES}. Waiting {wait}s…")
            time.sleep(wait)
            continue

        # Any other non-2xx status is a hard error
        resp.raise_for_status()

        data     = resp.json()
        candidate = data["candidates"][0]

        finish = candidate.get("finishReason", "")
        if finish not in ("STOP", ""):
            print(f"Warning: Gemini finishReason={finish!r} — response may be truncated.")

        raw = candidate["content"]["parts"][0]["text"].strip()
        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            findings = json.loads(raw)
            if not isinstance(findings, list):
                raise ValueError("Expected a JSON array.")
            print(f"Gemini returned {len(findings)} raw finding(s).")
            return findings
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: JSON parse failed: {e}\nRaw (first 500):\n{raw[:500]}")
            return [{
                "path": "", "line": 0,
                "category": "Correctness", "confidence": 80,
                "description": "Review model returned unparseable output. Raw:\n```\n" + raw[:2000] + "\n```",
                "suggestion": "",
            }]

    sys.exit(f"Gemini API still failing after {MAX_RETRIES} retries (last status: {resp.status_code}).")


# ---------------------------------------------------------------------------
# Split findings into inline comments vs general fallback
# ---------------------------------------------------------------------------
def build_review_payload(findings: list, patch_lines_map: dict):
    inline_comments = []
    general_findings = []

    for f in findings:
        path        = f.get("path", "")
        line        = f.get("line", 0)
        category    = f.get("category", "General")
        confidence  = f.get("confidence", 80)
        description = f.get("description", "")
        suggestion  = (f.get("suggestion") or "").strip()

        valid = patch_lines_map.get(path, set())
        # Snap to nearest valid hunk line within ±10 lines (catches Gemini off-by-one)
        anchored = snap_to_valid(line, valid) if path and line > 0 else 0

        body = f"**[{category}]** (Confidence: {confidence}%)\n\n{description}"
        if suggestion:
            body += f"\n\n```suggestion\n{suggestion}\n```"

        if path and anchored > 0:
            inline_comments.append({
                "path":  path,
                "line":  anchored,
                "side":  "RIGHT",
                "body":  body,
            })
        else:
            location = f"`{path}` line {line}" if path else "General"
            entry = f"### [{category}] — {location} (Confidence: {confidence}%)\n\n{description}"
            if suggestion:
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

    diff_context, patch_lines_map = build_diff_context(repo, files, head_sha)
    if not diff_context.strip():
        print("No reviewable files — skipping.")
        return

    print(f"Calling {GEMINI_MODEL}…")
    findings = call_gemini(pr_title, pr_body, diff_context)

    # Belt-and-braces: enforce confidence threshold
    findings = [f for f in findings if f.get("confidence", 0) >= 80]
    print(f"After confidence filter: {len(findings)} finding(s).")

    inline_comments, general_findings = build_review_payload(findings, patch_lines_map)
    print(f"Inline: {len(inline_comments)}, General fallback: {len(general_findings)}")

    # Build the top-level review body
    if not findings:
        verdict = "✅ **No high-confidence issues found — looks good to merge.**"
    else:
        count = len(inline_comments)
        verdict = f"⚠️ **{count} inline finding(s) — see comments on the changed lines below.**"

    summary_parts = [
        COMMENT_MARKER,
        "## 🤖 Maulome PR Review",
        f"*Model: `{GEMINI_MODEL}`*\n",
        verdict,
    ]

    if general_findings:
        summary_parts.append("\n---\n### 📋 General Findings\n")
        summary_parts.extend(general_findings)

    summary_parts.append(
        "\n---\n*Powered by [Maulome Review Bot](https://github.com/skyspec28/Maulome-Review-bot-)*"
    )

    dismiss_existing_summary(repo, pr_number)
    post_pr_review(repo, pr_number, head_sha, "\n".join(summary_parts), inline_comments)


if __name__ == "__main__":
    main()
