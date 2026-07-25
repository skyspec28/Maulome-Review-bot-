#!/usr/bin/env python3
"""
Kimi PR Reviewer — powered by OpenRouter.
Fetches the PR diff/files, sends them to the configured model, then
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
GITHUB_TOKEN        = os.environ["GITHUB_TOKEN"]
OPENROUTER_API_KEY  = os.environ["OPENROUTER_API_KEY"]
KIMI_MODEL          = os.environ.get("KIMI_MODEL", "moonshotai/kimi-k2")
MAX_FILE_CHARS      = int(os.environ.get("MAX_FILE_CHARS",  "8000"))
MAX_TOTAL_CHARS     = int(os.environ.get("MAX_TOTAL_CHARS", "60000"))

GITHUB_API          = "https://api.github.com"
OPENROUTER_API      = "https://openrouter.ai/api/v1/chat/completions"
COMMENT_MARKER      = "<!-- kimi-pr-reviewer -->"

# Files to skip (add patterns as needed)
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
    """Read PR metadata from the GITHUB_EVENT_PATH payload."""
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
        status   = f["status"]          # added / modified / removed / renamed
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
# OpenRouter call
# ---------------------------------------------------------------------------
def call_openrouter(pr_title, pr_body, diff_context):
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
        "model": KIMI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/skyspec28/Maulome-Review-bot-",
        "X-Title": "Maulome PR Reviewer",
    }
    resp = requests.post(OPENROUTER_API, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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

    print(f"Calling {KIMI_MODEL} via OpenRouter …")
    review_text = call_openrouter(pr_title, pr_body, diff_context)

    body = (
        f"{COMMENT_MARKER}\n"
        f"## 🤖 Maulome PR Review\n"
        f"*Model: `{KIMI_MODEL}`*\n\n"
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
