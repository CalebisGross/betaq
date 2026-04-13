# Git Safety

## Branch Workflow

- Remote: `origin` (https://github.com/CalebisGross/betaq.git)
- Primary branch: `main`
- **All new work starts on a feature branch** -- never commit directly to `main`
- Branch naming: `feat/<description>`, `fix/<description>`
- Before branching: `git stash` (if dirty), `git pull origin main`, then `git checkout -b <branch>`
- **Before committing:** Run `git branch --show-current` to verify you're on the intended branch. Bash tool does not persist shell state -- a prior `git checkout` may not have taken effect.
- **All changes go through a PR** -- push the branch, open a PR with `gh pr create`, get it reviewed
- **Closing issues:** When a PR resolves a GitHub issue, comment on the issue with a reference to the PR before or after closing it. Never close issues silently.

## Forbidden Operations

Enforced by `.claude/hooks/protect-git.sh` and `.claude/hooks/no-secrets.sh`:

- `git push --force` / `git push -f` -- destroys remote history
- `git reset --hard` -- destroys local changes
- `git clean -f` -- permanently deletes untracked files
- `git checkout .` / `git restore .` -- discards all unstaged changes
- Staging `.env`, `credentials`, `settings.local.json`

## Commit Messages (Conventional Commits)

- `feat: ...` -- new feature
- `fix: ...` -- bug fix
- `docs: ...` -- documentation only
- `refactor: ...` -- code change, no behavior change
- `test: ...` -- tests only
- `chore: ...` -- maintenance

Rules:

- Short, direct subject line describing the change
- Body for context when non-obvious
- No issue-closing keywords in commit messages unless explicitly asked
- Use Co-Authored-By for Claude contributions

## Secrets

- `settings.local.json` contains machine-specific permissions -- NEVER commit
- `.env` files -- gitignored
- Never include API tokens in commit messages or code
