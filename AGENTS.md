# Agent Instructions

## Project Context

This repository is an educational, from-scratch PyTorch implementation of an
end-to-end chat LLM, broadly aligned with nanochat's external pipeline and
metrics rather than its internal code. Build it in staged vertical slices:
local text -> byte tokenizer -> small decoder-only GPT -> pretrain/sample,
followed by ClimbMix-style data and regex byte-BPE, 10M-50M base training with
BPB/CORE evaluation, SFT and ChatCORE, shared inference for a CLI and local-only
web UI, optional W&B tracking, and finally performance and architecture
experiments. The initial target is one process on a single RTX 3090. Favor
correctness, simple and testable educational code, reproducible
configs/checkpoints, and always-available local JSONL metrics. See
`plans/pre-training.md` for the detailed roadmap, phase ordering, acceptance
criteria, and proposed tickets; consult the relevant section before
implementing a bead.

## Default Bead-to-PR Workflow

Unless the user explicitly requests a different scope, complete one bead per
review cycle:

1. Run `bd prime`, inspect `git status`, check existing in-progress work, and
   run `bd ready`. Resume relevant in-progress work instead of duplicating it.
   If no ready bead exists, stop and ask the user what to do next; do not create
   a bead from the roadmap on your own.
2. Select the next logical ready bead by respecting dependencies, roadmap phase
   order, priority, and the smallest coherent step toward the next vertical
   slice. If equally valid choices would change project direction, ask the user.
   Review it with `bd show <id>` and claim it with `bd update <id> --claim`.
3. Refresh the remote default branch, then create a dedicated git worktree and
   bead-specific branch from it. Keep one bead per worktree/branch and do not
   implement the bead in the primary checkout.
4. Use red/green/refactor TDD for each behavior:
   - **Red:** add the smallest test that expresses the acceptance criterion and
     run it to confirm it fails for the expected reason.
   - **Green:** write the minimum implementation that makes the test pass.
   - **Refactor:** improve the design while keeping tests green, then repeat.
   For documentation or other work where an automated test is not meaningful,
   use the closest deterministic validation available.
5. Run focused tests throughout development. Before publishing, run all
   relevant unit tests, linters, type checks, and builds documented by the
   repository.
6. Commit only the bead's scoped changes, push its branch, and open a pull
   request that references the bead ID and includes the change summary and test
   evidence. This workflow authorizes committing, pushing the bead branch, and
   creating or updating its PR without asking again. It does not authorize
   pushing the default branch, merging the PR, or running `bd dolt push`.
7. Run the integration test suite against the branch/PR when one exists and is
   applicable. If it fails, diagnose it, fix in-scope failures, push the update,
   and rerun it before calling the PR review-ready. If no integration test
   exists, say so explicitly in the handoff.
8. Once implementation and validation are complete, update the bead with
   completion notes using non-interactive `bd update` flags. Preserve useful
   existing notes and include the concise change summary, branch or commit,
   validation and integration results, PR URL, and any risks or follow-up. Leave
   the bead `in_progress` while it awaits review. Do not equate an open PR with
   merged work; close the bead only after confirming that the PR was merged.
9. Hand off with the bead ID, concise change summary, validation results
   (including integration status), PR URL, and any risks or follow-up. Stop for
   the user to review and merge; do not merge the PR yourself.

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

> **Architecture in one line:** Issues live in a local Dolt database
> (`.beads/dolt/`); cross-machine sync uses `bd dolt push/pull` (a
> git-compatible protocol), stored under `refs/dolt/data` on your git
> remote — separate from `refs/heads/*` where your code lives.
> `.beads/issues.jsonl` is a passive export, not the wire protocol.
>
> See .beads/SYNC_CONCEPTS.md for the one-screen overview and anti-patterns (don't treat JSONL as the source of truth; don't `bd import` during normal operation; don't
> reach for third-party Dolt hosting before trying the default).



## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```



## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**

```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**

- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var



## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```



### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See [https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md) for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.



## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
  ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
  ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**

- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.



## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```



### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See .beads/SYNC_CONCEPTS.md for details and anti-patterns.
