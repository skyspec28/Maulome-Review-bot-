#!/usr/bin/env python3
"""
Maulome PR Reviewer — powered by Google Gemini.
Fetches the PR diff/files, sends them to Gemini, then
posts (or edits) a single review comment on the PR.
"""

import os
import re
import sys
import json
import requests

# ---------------------------------------------------------------------------
# Configuration (override via env vars in the workflow)
# ---------------------------------------------------------------------------
GITHUB_TOKEN    = os.environ["GITHUB_TOKEN"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
MAX_FILE_CHARS  = int(os.environ.get("MAX_FILE_CHARS",  "8000"))
MAX_TOTAL_CHARS = int(os.environ.get("MAX_TOTAL_CHARS", "60000"))

GITHUB_API      = "https://api.github.com"
GEMINI_API      = (
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
                content = raw[:MAX_FILE_CHARS]
                if len(raw) > MAX_FILE_CHARS:
                    content += f"\n... [truncated — {len(raw) - MAX_FILE_CHARS} chars omitted]"

        chunk = f"### `{filename}` ({status})\n"
        if patch:
            chunk += f"**Diff:**\n```diff\n{patch}\n```\n"
        if content:
            ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
            chunk += f"**Full file:**\n```{ext}\n{content}\n```\n"

        if total + len(chunk) > MAX_TOTAL_CHARS:
            parts.append(f"### `{filename}` — skipped (total size limit reached)\n")
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------
def call_gemini(pr_title, pr_body, diff_context):
    system_prompt = (
        "You are an expert code reviewer. You will be given a pull request title, "
        "description, and the changed files. Provide a thorough, constructive review:\n"
        "- Summarise what the PR does.\n"
        "- Highlight bugs, security issues, or logic errors.\n"
        "- Point out style / maintainability concerns.\n"
        "- Suggest improvements with concrete examples where helpful.\n"
        "- End with an overall verdict: ✅ Approve / ⚠️ Request changes / ❌ Major issues.\n"
        "Be concise but complete. Use markdown formatting."
    )
    user_content = (
        f"## PR Title\n{pr_title}\n\n"
        f"## PR Description\n{pr_body or '_No description provided._'}\n\n"
        f"## Changed Files\n{diff_context}"
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": user_content}]}
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
        },
    }

    url = f"{GEMINI_API}?key={GEMINI_API_KEY}"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


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
