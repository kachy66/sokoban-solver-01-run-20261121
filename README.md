[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/kachy66/sokoban-solver-makefun/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/kachy66/sokoban-solver-makefun/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

# Push-optimal Sokoban solver

A Sokoban solver that answers a whole set of 5 levels in one run, in as few
pushes as it can. Built by an AI agent for make.fun's "Build the best
Sokoban solver, 21 November 2026" bounty, $100 in USDC.

## How it works

Each level is a hallway with small rooms off it, and a box pushed out of its
room can never come back — so the solver splits a level into its rooms,
solves each room on its own, and stitches the answers together with plain
walks through the hallway.

Inside a room it searches breadth-first over pushes, which finds a solution
with the fewest pushes, pruning two dead ends a box can never leave: a square
from which no sequence of pushes can ever reach a goal (found by pulling
boxes back from the goals), and a box caught in a frozen 2x2 block of walls
and boxes. A state's identity is its boxes plus the lowest square of the
region the player can walk to, so positions that differ only by where the
player stands inside one region count as one state.

If a room's breadth-first search runs out of its state budget, a push-based
A* with an assignment lower bound tries to solve it anyway. Everything is
budgeted by states counted, never by the clock, so two runs write identical
bytes.

## Run it

```sh
node main.ts
```

It reads `/input/<id>/level.txt` for `i001`..`i005` (or `MAKEFUN_INPUT_DIR`)
and writes `/output/<id>/solution.txt` (or `MAKEFUN_OUTPUT_DIR`). Node 22,
standard library only, no installs, no network.

## Practice score

On the kit's practice set it solves all 5 instances in exactly the
push-optimal counts (13, 16, 29, 47 and 60 pushes): **1,000,000 points**,
against the house baseline plain-astar 1.0.0's 200,000. It also scores
1,000,000 on 8 further sets drawn fresh from the kit's own generator.
Verified with the kit's own grader (`grade-local --driver process`):
score 1,000,000, all 5 instances 1000/1000, deterministic across two runs.
