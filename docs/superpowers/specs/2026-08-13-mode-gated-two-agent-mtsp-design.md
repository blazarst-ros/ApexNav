# Mode-Gated Two-Agent MTSP Design

## Goal

Replace frontier claims as the normal multi-agent coordination mechanism with
mode-gated, centralized two-agent task assignment. The planner uses MTSP only
when both configured agents need a new target and their highest-priority
navigation modes are identical. Different modes remain independent because
their target classes are disjoint by design.

## Scope and fixed decisions

- The system has exactly two agents (`NUM_AGENTS == 2`).
- The MTSP objective is `MINMAX`: minimize the longer of the two assigned route
  lengths, rather than their total length.
- MTSP edge costs are A* path lengths. Semantic value and object confidence are
  eligibility gates, not additive edge-cost terms.
- Only candidates belonging to the common mode are included in an MTSP solve.
- Each solve produces one route per agent; only each route's first target is
  executed. Map, perception, object, or frontier changes trigger later
  rolling replanning.
- Normal planning does not read or write `Frontier2D::claimed_by_`. It may be
  retained solely as an assertion/debug signal that two agents were not given
  the same target.

## Mode gate

For a planning cycle, determine each agent's highest-priority executable mode
before assigning either agent a target. The existing priority ordering remains
authoritative: high-confidence object, over-depth object, active frontier,
suspicious object, dormant frontier, then extreme fallback.

The dispatcher has two branches:

1. If the two modes differ, run each agent's existing mode-specific planner
   independently. Claims are not used: targets from distinct modes are assumed
   different.
2. If modes match and both agents need replanning, build one shared candidate
   pool for that exact mode and solve a two-salesman MTSP.

If only one agent needs replanning, retain its independent planner. A joint
solve must never overwrite the other agent's active, still-valid target.

## Eligible candidate pools

| Common mode | MTSP pool | Required eligibility |
| --- | --- | --- |
| `SEARCH_BEST_OBJECT` | Current high-confidence object clusters only | Existing high-confidence criterion and a valid A* approach path |
| `SEARCH_OVER_DEPTH_OBJECT` | Current over-depth object cluster(s) only | Nonempty and A*-reachable |
| semantic frontier modes | Active frontiers selected by the existing semantic/Hybrid significance gate | Semantic eligibility, active status, A*-reachable |
| geometric frontier modes | Active reachable frontiers | Active status and A*-reachable |
| dormant/extreme modes | The corresponding fallback target class only | Existing fallback safety and validity checks |

No object type is mixed with any frontier type, and no high-confidence object
pool is mixed with suspicious or over-depth objects.

## Joint distance model

Let `R0` and `R1` be the current positions of agents 0 and 1, and let
`T1..Tn` be the filtered common candidate pool. The MTSP instance must retain
both starts:

`C(Rk, Ti) = AStarLength(Rk, Ti)` for each agent `k` and target `i`.

`C(Ti, Tj) = AStarLength(Ti, Tj)` for target-to-target transitions.

Unreachable edges are excluded at candidate-filter time, or represented by a
large finite penalty only when the solver still needs a complete matrix.
Euclidean distance is not used as the assignment cost.

The LKH parameter file must use `SALESMEN = 2` and
`MTSP_OBJECTIVE = MINMAX`. It must also request an MTSP-specific solution file
so routes can be decoded separately for agent 0 and agent 1. The existing
single-agent `TOUR_FILE` parser is insufficient for this purpose.

## Assignment and fallback behavior

- With fewer than two eligible targets, do not run MTSP. For one target, give
  it to the agent with the shorter valid A* path; the other agent receives no
  assignment from that shared pool and follows its normal fallback behavior.
- Reject malformed solver output, a missing route, duplicate target ownership,
  or a target whose path is no longer valid. Fall back to independent planning
  for that cycle.
- A successful MTSP assignment writes only one first target and one path per
  agent. It does not commit the full MTSP route as a persistent plan.
- Existing object-reached, stuck, dormant-frontier, and map-update behavior
  remains responsible for invalidating targets and requesting replanning.

## Architecture boundary

Introduce a centralized, testable assignment component at the FSM/planning
boundary. It consumes both agents' replan requests, modes, positions, and a
shared candidate pool, then produces zero, one, or two per-agent assignments.
The existing single-agent policy functions remain responsible for computing
mode-specific candidates and for independent fallback planning.

The LKH wrapper must expose a structured two-route result to the planner
instead of requiring the planner to interpret a flattened `TOUR_FILE`.

## Verification

- Unit tests prove that different modes choose independent planning with no
  claim read/write.
- Unit tests prove that matching semantic, geometric, high-confidence-object,
  and over-depth-object modes form only their corresponding candidate pools.
- Unit tests prove that a matching-mode joint request uses both agent starts,
  `SALESMEN = 2`, and `MTSP_OBJECTIVE = MINMAX`.
- Unit tests prove that one-target, invalid-edge, malformed-solver-output, and
  duplicate-target cases fall back safely.
- A parser test verifies that the two LKH MTSP routes are mapped to distinct
  agent assignments and only each route's first target is dispatched.
- Run targeted tests, the project test suite available in this workspace, a
  ROS/catkin build when the local environment is available, and `git diff
  --check`.
