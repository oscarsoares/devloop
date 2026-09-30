# devloop

An always-on executor. Per tick, against one repository:

```
any actionable PR?    -> review it    -> back to the top
else any ready issue? -> develop it   -> back to the top
else                  -> nothing to do, stop
```

Reviews come before new development, because the next task may depend on an open PR landing.
Opening a PR is not an exit condition: the PR the loop just produced is reviewable work.

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
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m pyright
.venv/Scripts/python.exe -m ruff check .
```

All three are expected to pass with zero findings before a commit.

## Status

Ported and tested: stream parsing, the PR classifier, backlog triage, the driver seam, the
CLI driver, GitHub reads behind their own interface, and two commands.

```bash
devloop status oscarsoares/altrus        # what the loop sees and would do; writes nothing
devloop run "summarise the open PRs"     # one prompt, streaming events and cost
```

`status` is read-only by construction, which is what makes the rest inspectable: you can see
the decision before anything acts on it.

State is durable, in SQLite beside the user's other tool state (`~/.devloop/state.db`). It
holds the three things that must survive a tick ending and do not exist on GitHub: review
rounds per PR, open escalations, and cycle history with spend.

`tick` runs one pass of the loop with its budgets and writes labels and comments through `gh`;
it is a dry run unless `--execute` is given.

Not yet built: per-repository configuration and the scheduler.
