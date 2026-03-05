---
description: |
  Reviews a pull request against the nocapchicken ML project rubric.
  Posts a structured review comment on the PR.

strict: false

engine:
  id: copilot
  model: claude-sonnet-4

on:
  workflow_dispatch:
    inputs:
      pr_number:
        description: "PR number to review"
        required: true

permissions: read-all

safe-outputs:
  add-comment:

tools:
  github:
    toolsets: [pull_requests, repos]

timeout-minutes: 10
---

# PR Review

Review PR #${{ github.event.inputs.pr_number }} for the nocapchicken ML project against `REQUIREMENTS_CHECKLIST.md`.

Check:
- **Rubric compliance** — flag violations by ID (e.g. CQ5, GIT6)
- **Code quality** — functions/classes only, docstrings on public functions, AI attribution (CQ5/CQ6)
- **Security** — no `.env` committed (GIT6), no hardcoded secrets
- **ML correctness** — no training in `app/` (APP1), feature vector matches `build_features.py`
- **App stability** — Flask routes handle errors and API failures gracefully

Deduction triggers: experiments not tied to a decision (−5), vague error analysis (−5), PR without summary (−5), "LGTM" review (−2).

Post a comment on the PR with this format:
- **Summary** (2–3 sentences)
- **Checklist items affected**
- **Issues** (blocking, numbered)
- **Suggestions** (non-blocking, numbered)
- **Verdict**: Approve / Request Changes / Comment
