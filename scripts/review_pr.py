#!/usr/bin/env python3
"""
Maulome PR Reviewer — powered by Google Gemini.

Workflow:
1. Fetch PR files/diff via GitHub API.
2. Build diff context with line-numbered file content so Gemini can cite exact lines.
3. Send to Gemini with a structured per-finding prompt.
4. Post (or edit) a single PR issue comment with:
   - Summary of what the PR does.
   - Per-finding blocks: file + line + category + explanation + concrete fix snippet.
   - Verdict (✅ / ⚠️ / ❌).
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


def find_existing_comment(repo, pr_number):
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    resp = requests.get(url, headers=gh_headers())
    resp.raise_for_status()
    for c in resp.json():
        if COMMENT_MARKER in c.get("body", ""):
            return c["id"]
    return None


def post_comment(repo, pr_number, body):
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    resp = requests.post(url, headers=gh_headers(), json={"body": body})
    resp.raise_for_status()
    print(f"Posted new review comment (id={resp.json()['id']}).")


def edit_comment(repo, comment_id, body):
    url = f"{GITHUB_API}/repos/{repo}/issues/comments/{comment_id}"
    resp = requests.patch(url, headers=gh_headers(), json={"body": body})
    resp.raise_for_status()
    print(f"Updated existing review comment (id={comment_id}).")


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------
def should_skip(filename):
    return any(re.search(pat, filename) for pat in SKIP_PATTERNS)


def build_diff_context(repo, files, head_sha):
    """
    Build the text context sent to Gemini.
    Each file gets:
      - The raw unified diff (so Gemini knows what actually changed).
      - The full post-merge file content with line numbers in 'NNNNN | code' format,
        so Gemini can cite exact line numbers in its findings.
    """
    parts = []
    total = 0

    for f in files:
        filename = f["filename"]
        status   = f["status"]
        if should_skip(filename):
            continue

        patch = f.get("patch", "")
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

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Gemini API call — returns structured markdown review text
# ---------------------------------------------------------------------------
def call_gemini(pr_title, pr_body, diff_context) -> str:
    system_prompt = (
        "You are an expert code reviewer. You will be given a pull request title, "
        "description, and changed files. Each file's full content is shown with "
        "line numbers in the format 'NNNNN | code' — use these exact numbers when citing lines. "
        "Diffs (unified format) are also shown so you know what actually changed.\n\n"
        "Analyze the changes for:\n"
        "1. Correctness — logical errors, edge cases, error handling, state inconsistency, race conditions.\n"
        "2. Security — vulnerabilities, credential leaks, injection risks, improper input validation.\n"
        "3. Performance — inefficient queries, memory leaks, slow algorithms, redundant operations.\n"
        "4. Breaking Changes — backward-compat issues, API signature changes, schema changes.\n\n"
        "RULES:\n"
        "- Only report findings with confidence >= 80%.\n"
        "- Only report issues in code that was actually added/changed in the diff — not pre-existing code you're just seeing for context.\n"
        "- No style/formatting nitpicks.\n"
        "- For EVERY finding, you MUST give: the exact file path, the exact line number "
        "(from the numbered file content), a one-line issue title, a short explanation of why "
        "it's a problem, and a concrete fix — either a short corrected code snippet or precise steps to fix it.\n\n"
        "Output format — repeat this block per finding, nothing else outside it:\n\n"
        "### [N]. <Short issue title>\n"
        "**File:** `path/to/file.py` — **Line:** <number>\n"
        "**Category:** <Correctness|Security|Performance|Breaking Change> — **Confidence:** <NN>%\n"
        "**Issue:** <1-3 sentence explanation>\n"
        "**Fix:**\n```<lang>\n<corrected snippet or precise fix steps>\n```\n\n"
        "Start with a 2-3 sentence '### Summary' of what the PR does, then list findings under "
        "'### Detailed Findings', then end with '### Verdict' "
        "(✅ Approve / ⚠️ Request changes / ❌ Major issues). "
        "If there are no findings above 80% confidence, say so explicitly under Detailed Findings."
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

        # Warn if the model was cut off before finishing
        finish_reason = candidate.get("finishReason", "")
        if finish_reason not in ("STOP", ""):
            print(f"Warning: Gemini finishReason={finish_reason!r} — response may be truncated.")

        return candidate["content"]["parts"][0]["text"]

    sys.exit(f"Gemini API still returning 429 after {MAX_RETRIES} retries.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    repo, pr_number, base_sha, head_sha, pr_title, pr_body = get_pr_info()
    print(f"Reviewing PR #{pr_number} in {repo}  ({base_sha[:7]}..{head_sha[:7]})")

    files = get_pr_files(repo, pr_number)
    print(f"Files changed: {len(files)}")

    diff_context = build_diff_context(repo, files, head_sha)
    if not diff_context.strip():
        print("No reviewable files found — skipping.")
        return

    print(f"Calling {GEMINI_MODEL} via Google AI …")
    review_text = call_gemini(pr_title, pr_body, diff_context)

    body = (
        f"{COMMENT_MARKER}\n"
        f"## 🤖 Maulome PR Review\n"
        f"*Model: `{GEMINI_MODEL}`*\n\n"
        f"{review_text}\n\n"
        f"---\n*Powered by [Maulome Review Bot](https://github.com/skyspec28/Maulome-Review-bot-)*"
    )

    comment_id = find_existing_comment(repo, pr_number)
    if comment_id:
        edit_comment(repo, comment_id, body)
    else:
        post_comment(repo, pr_number, body)


if __name__ == "__main__":
    main()
