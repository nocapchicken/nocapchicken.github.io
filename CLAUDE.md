# CLAUDE.md — nocapchicken

This file is loaded automatically by Claude Code in every session.
All instructions here are mandatory and override default behavior.

---

## Project Context

Academic ML module project: predict NC restaurant food-safety grades from
crowdsourced review data (Yelp + Google Places). Final deliverables include
a written report, a live deployed Flask app, and an in-class pitch.

---

## REQUIREMENTS_CHECKLIST.md is the source of truth

**Before writing any code, always read `REQUIREMENTS_CHECKLIST.md`.**

Every task you perform must be evaluated against the checklist. When you
complete work that advances a checklist item, say which item ID(s) it satisfies
(e.g. "this satisfies M7 and CQ5"). When you notice a checklist item is at risk
or being violated, call it out explicitly before proceeding.

---

## Hard rules (never violate these)

### Code quality
- All code must live inside functions or classes. No loose executable code
  outside `if __name__ == "__main__"` guards. (CQ1, CQ2)
- Every public function must have a docstring. (CQ4)
- If you write or assist with code in a file, add an AI attribution comment
  at the top of that file with a link to this session. (CQ5)
  Example: `# AI-assisted (Claude Code, claude.ai) — https://claude.ai`
- Use descriptive variable names. Never use single-letter names outside
  loop indices or math. (CQ3)

### Comments and docs style
- **Be concise.** Docstrings state what's non-obvious; never restate the code.
- Inline comments only where logic isn't self-evident.
- PR review comments: flag the issue and its checklist ID — skip lengthy
  explanations. Team members can ask Claude directly for elaboration.

### Security
- Never commit or suggest committing `.env`, API keys, or secrets. (GIT6)
- Never hardcode credentials in source files.
- Never commit large data files or trained model binaries. (GIT7)

### App integrity
- `app/` is inference-only. Never add training logic there. (APP1)
- Feature construction in `app/inference.py` must stay in sync with
  `scripts/build_features.py`. Flag any drift.

### Git workflow
- All changes go through PRs — never suggest direct commits to `main`. (GIT3)
- Every PR must have a written Summary paragraph. Remind the user if they
  haven't written one. (GIT4)
- Substantive PR reviews are required — "LGTM" is not acceptable. (GIT5)

---

## Known grader deduction triggers

These patterns caused point losses in a prior cohort (83/100). Actively
prevent them:

| Risk | Deduction | How to avoid |
|------|-----------|--------------|
| Experiment not connected to a modeling decision | -5 | Every experiment must end with a concrete change or validation of a design choice (EX5) |
| Vague error analysis mitigations | -5 | Each of the 5 mispredictions needs a *specific* fix, not "collect more data" (R12) |
| PRs without descriptions | -5 | Remind user to fill in PR Summary before merging (GIT4) |
| "LGTM"-only reviews | -2 | PR reviews must include at least one substantive comment (GIT5) |
| Future Work section buried | flag | Must be a clearly labelled top-level section in the report (R15) |

---

## Before every commit or PR

Run the `code-simplifier` agent on all files changed in the current branch
before staging a commit or opening a PR:

```
# invoke via Claude Code skill
code-simplifier
```

The agent refines code for clarity, consistency, and maintainability without
changing functionality. It focuses on recently modified files by default, so
running it right before a commit is the right time. Do not skip this step —
clean, readable code is part of the graded code quality criteria (CQ1–CQ4).

---

## Jupyter notebooks

Allowed **only** in `notebooks/`. They are for exploration and will not be
graded. Production code goes in `scripts/` and `app/`. Never import from
a notebook in any other module.

---

## When asked to help with the written report

The report must include all sections R01–R17 from `REQUIREMENTS_CHECKLIST.md`.
Always check which sections are still unchecked before drafting content.
Pay special attention to:
- R04 (metric justification is marked "critical" by the grader)
- R10–R12 (error analysis — 5 specific cases, concrete mitigations)
- R15 (Future Work must be a clearly visible section)

---

## When asked to help with experiments

Ensure the experiment satisfies EX1–EX6. Specifically, EX5 is the most
commonly missed: the experiment must directly inform or validate a decision
already made (or to be made) in the modeling pipeline.
