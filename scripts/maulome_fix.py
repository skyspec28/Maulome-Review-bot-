#!/usr/bin/env python3
"""
Maulome Fixer — powered by Google Gemini.

Workflow:
1. Parse the GitHub comment event (pull_request_review_comment or issue_comment).
2. Fetch the comment thread context if it's a reply to an inline finding.
3. Read the relevant files directly from the checked-out workspace.
4. Call Gemini to generate the corrected file content.
5. Write the corrected file to disk, commit, and push back to the PR branch.
"""

import os
import re
import sys
import json
import time
import requests
import subprocess

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GITHUB_TOKEN    = os.environ["GITHUB_TOKEN"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
MAX_RETRIES     = int(os.environ.get("MAX_RETRIES", "5"))

GITHUB_API      = "https://api.github.com"
GEMINI_API_URL  = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

SKIP_PATTERNS = [
    r"\.lock$", r"package-lock\.json$", r"yarn\.lock$",
    r"\.min\.(js|css)$", r"dist/", r"build/", r"__pycache__/", r"\.pyc$",
    r"\.png$", r"\.jpg$", r"\.jpeg$", r"\.gif$", r"\.svg$", r"\.ico$",
    r"\.pdf$", r"\.zip$", r"\.tar$",
]


def should_skip(filename):
    import re
    return any(re.search(pat, filename) for pat in SKIP_PATTERNS)


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------
def gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_parent_comment(repo, comment_id):
    url = f"{GITHUB_API}/repos/{repo}/pulls/comments/{comment_id}"
    resp = requests.get(url, headers=gh_headers())
    if resp.ok:
        return resp.json()
    return None


def get_pr_files(repo, pr_number):
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files"
    resp = requests.get(url, headers=gh_headers())
    resp.raise_for_status()
    return resp.json()


def post_reply_comment(repo, pr_number, comment_id, body, is_review_comment=True):
    """Post a reply to the comment thread to let the user know the outcome."""
    if is_review_comment:
        url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/comments/{comment_id}/replies"
        payload = {"body": body}
    else:
        url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
        payload = {"body": body}
    
    resp = requests.post(url, headers=gh_headers(), json=payload)
    if resp.ok:
        print("Posted reply comment.")
    else:
        print(f"Failed to post reply: {resp.text}")


# ---------------------------------------------------------------------------
# Git / Shell helpers
# ---------------------------------------------------------------------------
def run_command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command {' '.join(args)} failed:")
        print(f"Stdout:\n{result.stdout}")
        print(f"Stderr:\n{result.stderr}")
    return result


def commit_and_push(branch_name, file_path, commit_message):
    run_command(["git", "config", "user.name", "maulome-bot"])
    run_command(["git", "config", "user.email", "bot@maulome.dev"])
    
    # Stage the file
    run_command(["git", "add", file_path])
    
    # Commit
    commit_res = run_command(["git", "commit", "-m", commit_message])
    if "nothing to commit" in commit_res.stdout or "nothing added to commit" in commit_res.stdout:
        print("Nothing to commit (file is already up to date).")
        return False
        
    # Push back to the head branch
    push_res = run_command(["git", "push", "origin", f"HEAD:{branch_name}"])
    if push_res.returncode == 0:
        print("Successfully pushed commit back to PR branch.")
        return True
    return False


# ---------------------------------------------------------------------------
# Gemini call — returns structured JSON with the updated file content
# ---------------------------------------------------------------------------
def call_gemini_for_fix(file_path, file_content, line_number, instruction, original_comment=None) -> dict:
    system_prompt = (
        "You are an expert code developer helper. Your job is to take a source file, "
        "a specific line context, a user instruction (which may be a request to fix a bug, "
        "apply a suggestion, or modify logic), and output the **complete updated file content**.\n\n"
        "OUTPUT RULES — follow strictly:\n"
        "- Return ONLY a raw JSON object. No markdown fences, no prose, no explanation outside the JSON.\n"
        "- The JSON object must contain exactly these two keys:\n"
        '  {"path": str, "content": str}\n'
        "- `path`: the exact path of the file being updated.\n"
        "- `content`: the FULL updated content of the file. Do not truncate, do not use placeholders like '// ... rest of code'. "
        "Return the entire file from start to finish with the requested changes applied.\n"
        "- Ensure the indentation, imports, and surrounding structure of the file are fully preserved.\n"
    )

    context_str = f"File Path: {file_path}\n"
    if line_number:
        context_str += f"Target Line: {line_number}\n"
    if original_comment:
        context_str += f"Original Bot Finding/Suggestion:\n{original_comment}\n"

    user_content = (
        f"## Code Context\n{context_str}\n\n"
        f"## User Instruction\n{instruction}\n\n"
        f"## Original File Content\n```\n{file_content}\n```"
    )

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {
            "temperature": 0.1,
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
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or "content" not in data:
                raise ValueError("Expected JSON object with 'content' key.")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: JSON parse failed: {e}\nRaw (first 500):\n{raw[:500]}")

    sys.exit(f"Gemini API still failing after {MAX_RETRIES} retries (last status: {resp.status_code}).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path or not os.path.exists(event_path):
        sys.exit("GITHUB_EVENT_PATH not set or file missing.")
        
    with open(event_path) as f:
        event = json.load(f)

    # Determine event type
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    print(f"Running fix trigger for event: {event_name}")

    repo = event["repository"]["full_name"]
    comment = event["comment"]
    comment_body = comment["body"].strip()
    comment_id = comment["id"]

    # Extract user instruction (strip out the bot call prefix e.g. @maulome fix)
    instruction = re.sub(r"^@maulome\s+fix\b:?\s*", "", comment_body, flags=re.IGNORECASE).strip()
    
    # We need to know the head branch name to push to
    # Fetch from PR details since issue_comment/pull_request_review_comment events
    # contain the pull_request object or PR number
    pr_number = 0
    head_branch = ""
    is_review_comment = False

    if "pull_request" in event:
        pr_number = event["pull_request"]["number"]
        head_branch = event["pull_request"]["head"]["ref"]
        is_review_comment = True
    elif "issue" in event and event["issue"].get("pull_request"):
        pr_number = event["issue"]["number"]
        # Fetch branch name from GitHub API since issue event doesn't have head ref
        pr_url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
        resp = requests.get(pr_url, headers=gh_headers())
        resp.raise_for_status()
        head_branch = resp.json()["head"]["ref"]

    if not pr_number or not head_branch:
        sys.exit("Could not resolve PR number or head branch name.")

    print(f"PR Number: #{pr_number}, Branch: {head_branch}")

    target_file = ""
    line_number = 0
    original_comment_body = ""

    # Check if this is an inline review comment reply
    if is_review_comment and "path" in comment:
        target_file = comment["path"]
        line_number = comment.get("line") or comment.get("original_line") or 0
        
        # If it's a reply, fetch the original review comment thread to get the bot's suggestion
        in_reply_to_id = comment.get("in_reply_to_id")
        if in_reply_to_id:
            parent = get_parent_comment(repo, in_reply_to_id)
            if parent:
                original_comment_body = parent.get("body", "")

    # If it is a general PR comment (issue_comment), try to parse the file path from instructions
    # or look at the list of files changed in the PR.
    if not target_file:
        files = get_pr_files(repo, pr_number)
        changed_paths = [f["filename"] for f in files if not should_skip(f["filename"])]
        
        # Check if the user specified one of the changed files in the comment body
        for path in changed_paths:
            basename = os.path.basename(path)
            if basename in instruction or path in instruction:
                target_file = path
                break
        
        # Default to the first changed file if none is specified
        if not target_file and changed_paths:
            target_file = changed_paths[0]

    if not target_file or not os.path.exists(target_file):
        post_reply_comment(
            repo, pr_number, comment_id,
            f"❌ Could not locate the target file `{target_file}` in the workspace.",
            is_review_comment
        )
        sys.exit(f"Target file {target_file} not found locally.")

    print(f"Target File: {target_file}, Line: {line_number}")

    # Read current file content from the locally checked out workspace
    with open(target_file, "r", encoding="utf-8", errors="replace") as f:
        file_content = f.read()

    # If user just said "@maulome fix" with no instruction, and there is a suggestion block in the parent comment,
    # let's try to extract and apply the suggestion block directly!
    suggestion_applied = False
    if not instruction and original_comment_body:
        sug_match = re.search(r"```suggestion\n(.*?)\n```", original_comment_body, re.DOTALL)
        if sug_match:
            suggestion_content = sug_match.group(1)
            print("Found native suggestion block in parent comment. Applying directly...")
            
            # Simple replacement logic: find the line and replace it
            lines = file_content.splitlines()
            if line_number > 0 and line_number <= len(lines):
                # Replace the target line (preserving correct zero-indexing)
                lines[line_number - 1] = suggestion_content
                new_content = "\n".join(lines) + ("\n" if file_content.endswith("\n") else "")
                
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(new_content)
                suggestion_applied = True

    if not suggestion_applied:
        # Query Gemini to generate the corrected file
        print("Calling Gemini to generate fix...")
        result = call_gemini_for_fix(
            target_file, file_content, line_number,
            instruction or "Apply the suggested fix", original_comment_body
        )
        
        new_content = result.get("content", "")
        if not new_content.strip():
            post_reply_comment(
                repo, pr_number, comment_id,
                "❌ Gemini returned empty content for the fix.",
                is_review_comment
            )
            sys.exit("Empty content returned from Gemini.")

        # Write the updated content to disk
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(new_content)

    # Commit and push
    commit_msg = f"fix: address Maulome review comment on {target_file}"
    if line_number:
        commit_msg += f" (line {line_number})"
        
    pushed = commit_and_push(head_branch, target_file, commit_msg)
    
    if pushed:
        post_reply_comment(
            repo, pr_number, comment_id,
            f"✅ Successfully generated and pushed the fix to branch `{head_branch}`!",
            is_review_comment
        )
    else:
        post_reply_comment(
            repo, pr_number, comment_id,
            "⚠️ No changes detected or commit failed. Is the fix already applied?",
            is_review_comment
        )


if __name__ == "__main__":
    main()
