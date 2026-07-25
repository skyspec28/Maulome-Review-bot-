# Kimi PR Reviewer (OpenRouter, reusable-workflow setup)

Central repo holds the logic; every target repo just references it with a
tiny stub. Update the model, provider, or prompt once in the central repo,
and it applies everywhere on the next PR.

## Repo layout (this bundle)

```
scripts/review_pr.py                          → goes in the CENTRAL repo
.github/workflows/reusable-review.yml          → goes in the CENTRAL repo
stub-for-target-repos/.github/workflows/kimi-review.yml  → goes in EVERY OTHER repo
```

## Setup

### 1. Create the central repo
e.g. `skyspec28/Maulome-Review-bot-`. Push `scripts/review_pr.py` and
`.github/workflows/reusable-review.yml` to it, keeping the same paths.

If you rename the repo or your username differs, update the `repository:`
and `uses:` lines in both `reusable-review.yml` and the stub to match.

### 2. Get an OpenRouter API key
https://openrouter.ai/keys — add credit to the account (OpenRouter is
pay-as-you-go, not subscription).

### 3. Add the secret
- **If your repos are under one GitHub org:** add `OPENROUTER_API_KEY` as an
  org-level secret, scoped to the repos you want. `secrets: inherit` in the
  stub picks it up automatically — set once, done everywhere.
- **If they're personal (non-org) repos:** GitHub secrets don't cross repos
  automatically, so add `OPENROUTER_API_KEY` to each target repo
  individually (Settings → Secrets and variables → Actions).

### 4. Drop the stub into each target repo
Copy `stub-for-target-repos/.github/workflows/kimi-review.yml` into
`.github/workflows/kimi-review.yml` in `zuba-africa` and any other repo you
want reviewed. That's the entire footprint per repo — one file, six lines.

### 5. Test
Open a PR in a target repo, confirm a comment posts. Push again, confirm it
edits the same comment instead of adding a new one.

## Changing the model later
Edit `KIMI_MODEL` in `reusable-review.yml` (currently `moonshotai/kimi-k2`)
to any OpenRouter model string — e.g. `anthropic/claude-sonnet-5`,
`deepseek/deepseek-v3`, `openai/gpt-5`. One edit, applies to every repo
using the stub on their next PR. No new API key needed since it's all
routed through OpenRouter.

## Tuning
- `MAX_FILE_CHARS` / `MAX_TOTAL_CHARS` env vars in `reusable-review.yml`
  control how much file content gets sent per review (cost control).
- `SKIP_PATTERNS` in `review_pr.py` controls which files are excluded.

## Note on the checkout step
`reusable-review.yml` does two checkouts: the calling repo (to read its PR
diff/files) and this central repo (to get `review_pr.py`). If you make the
central repo private, the token used for that second checkout needs read
access to it — either keep the central repo public, or use a PAT with
cross-repo read access instead of the default `GITHUB_TOKEN`.
