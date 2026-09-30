# devloop

An always-on executor. Per tick, against one repository:

```
any actionable PR?    -> review it    -> back to the top
else any ready issue? -> develop it   -> back to the top
else                  -> nothing to do, stop
```

Reviews come before new development, because the next task may depend on an open PR landing.
Opening a PR is not an exit condition: the PR the loop just produced is reviewable work.

## Install

Prerequisites:

- **Python 3.12 or newer.**
- **[`gh`](https://cli.github.com/), authenticated** (`gh auth login`). All GitHub reads and
  writes go through it, so devloop holds no token of its own.
- **[`claude`](https://docs.claude.com/en/docs/claude-code) CLI, signed in.** The default
  driver spawns it.
- A local checkout of the repository to work on, passed as `--cwd`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"    # .venv/bin/python on macOS and Linux
```

This installs a `devloop` command (or use `python -m devloop`).

## Usage

```bash
devloop status owner/repo          # what the loop sees and would do; writes nothing
devloop run "summarise the PRs"    # one prompt through Claude, streaming events and cost
devloop tick owner/repo            # one pass of the loop, as a dry run
devloop tick owner/repo --execute  # the same pass, for real
```

### `tick`

```
devloop tick REPO [--cwd DIR] [--state PATH] [--max-cycles N] [--max-minutes N]
                  [--max-cost USD] [--permission-mode MODE] [--execute]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `REPO` | required | `owner/name`, e.g. `oscarsoares/altrus` |
| `--cwd DIR` | current directory | Local checkout Claude works in |
| `--state PATH` | `~/.devloop/state.db` | SQLite database holding the loop's history |
| `--max-cycles N` | 6 | Reviews plus developments allowed in one tick |
| `--max-minutes N` | 150 | Wall-clock limit, checked between cycles |
| `--max-cost USD` | none | Stop once this much is spent in the tick |
| `--permission-mode MODE` | `acceptEdits` | Passed to `claude --permission-mode` |
| `--execute` | off | Act for real; without it the tick is a dry run |

**Without `--execute` nothing is written and Claude is not run.** The loop is handed a writer
that only records, so a dry run cannot touch GitHub by construction. It reports the decision
it would have made and stops. Start there, and start a first `--execute` with `--max-cycles 1`
on a repository you can afford to be wrong on.

**With `--execute`** a tick reviews and develops until a budget is reached or the work runs
out. The budgets are runaway guards, not limits on intent: what is left carries to the next
tick, and a cycle in flight is never cut off. The cost budget counts a cycle that reported no
cost as zero, so it can stop late, never early.

It ends with a summary:

```
Tick finished.
  PRs reviewed:        2
  Issues developed:    1
  Cost this tick:      $1.3400
  Cost, all recorded:  $4.8100 over 9 cycle(s)
  Stopped:             no actionable PR and no selectable issue
```

A cost Claude did not report is shown as "at least", never as zero. `tick` exits 1 when a
cycle failed, the directory does not exist, or `gh` or `claude` could not be run.

### What a review and a development do

**Review.** The loop loads the PR's title, description, diff and comments, and asks Claude to
find problems and suggest improvements. The reply must end with one line:

```
DECISION: approve | request_changes | block
```

The loop reads the last such line. Its reply is posted as a PR comment and the merge-gate
labels follow the decision:

| Decision | Labels on the PR | Also |
| --- | --- | --- |
| `approve` | `needs-review` added, `review-blocking` removed | |
| `request_changes` | `review-blocking` added, `needs-review` removed | |
| `block` | `review-blocking` added, `needs-review` removed | an escalation is recorded |

A reply with no readable `DECISION` is a failed round: it is recorded as spend but not counted
against the PR, nothing is posted, and the tick stops.

**Development.** The loop asks Claude to implement the issue on a branch named
`agent/issue-N-<slug>`, write tests, commit, push, and open a PR saying `Closes #N`. That
branch name and `Closes #N` are how the loop recognises that a PR already owns the issue.

### Limitations

> **`request_changes` and `block` do not fix anything.** The review prompt only reviews;
> nothing applies the changes it asks for. A PR labelled `review-blocking` counts as
> actionable, so it is reviewed again on later ticks, costing money each round and changing
> nothing, until it reaches the round limit (3) and is left for a human. A `block` records an
> escalation but does not stop those repeat reviews. Watch `devloop status` for PRs that keep
> coming back.

Also not built yet: per-repository configuration and a scheduler. Run `tick` yourself, or from
cron or Task Scheduler.

## What is stored

State lives in SQLite (`~/.devloop/state.db` unless `--state` says otherwise). It holds what
must survive a tick ending and does not exist on GitHub:

- **Per PR:** review rounds so far, cost accumulated, the last decision, and when it last
  acted. The round count is what enforces the per-PR limit; without it a PR whose gate stays
  held would be reviewed forever.
- **Per issue:** development attempts, cost accumulated, status (`in_progress`, `done` or
  `abandoned`) and when it last changed.
- **Cycles:** one row per Claude run with its cost and token counts. The measured cost per
  cycle is what the choice between the CLI and the Agent SDK waits on.
- **Escalations:** what is waiting on a human, shown by `devloop status`.

The per-PR and per-issue costs and the cycle costs are the same money seen two ways. The
totals the commands print come from the cycles, because only they can say when a figure is a
floor; do not add the two together.

## Why this exists

It replaces a PowerShell script that grew past what PowerShell should carry. Three of that
script's bugs were: a property read on an object that did not have it, a regex capture that
survived into the next loop iteration, and two API payload shapes that differ by one field
name. All three reached runtime. All three are what `pyright --strict` exists to catch, and
the checker rejected the first draft of this port for exactly that class of mistake.

## Design

**Decisions are pure functions over plain data.** `decide.py` takes `PullRequest` and `Issue`
values and returns verdicts. Nothing in it touches the network, so every judgement is tested
without one. That testability is the property the predecessor lacked.

**Claude sits behind one interface.** `drivers.ClaudeDriver` yields `Event`s; nothing above
it knows whether they came from a spawned CLI or from the Agent SDK. The two differ in cost —
the CLI runs on a Claude Code subscription, while the Agent SDK's documented auth is an API
key, billed per token. Choosing between them needs a measured cost per cycle we do not have
yet, and `--bare` becoming the default for `claude -p` will end the subscription path on its
own schedule. Behind an interface, that decision costs a driver, not a rewrite.

**The parser never raises.** A stream is a live process's stdout. A line the parser does not
recognise becomes `Unknown` and the run continues; a parser that raised would kill the tick
that produced the line.

**Writes are a separate interface from reads.** A dry run gets a writer that cannot write,
rather than a real one it is trusted not to call.

## Two judgements worth not tidying away

**A never-reviewed PR outranks pending CI.** A PR the loop just opened always has CI running,
and reading a diff does not depend on CI — deferring there would make the loop ignore the
work it just produced. Once reviewed, pending CI genuinely is the next signal, so it waits.
`TestPendingCiCutsBothWays` holds both halves.

**Bugs jump the queue, but only labelled ones.** A `bug` at P0 or P1 outranks every feature
milestone: a tenant leak is not something to build features on top of. Severity alone cannot
tell a defect from a feature, so the `bug` label is required — and a P3 bug does not jump.

## Develop

```bash
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m pyright
.venv/Scripts/python.exe -m ruff check .
```

All three are expected to pass with zero findings before a commit. No test spawns `gh` or
`claude`: both are replaced, so the suite runs offline.
