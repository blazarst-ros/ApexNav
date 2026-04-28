# Refactoring Workflow for Multi-Agent Zero-Shot Object Navigation

## 1. Goal

This workflow describes how to realize a multi-agent zero-shot object navigation system by refactoring the current ApexNav-style stack into an **occupancy-map-backed MR-DTG architecture**.

The intended final behavior is:

- multiple robots share geometric and semantic understanding,
- robots navigate toward unseen target objects in a zero-shot manner,
- occupancy maps remain the base for perception, collision checking, and visualization,
- MR-DTG becomes the high-level exploration, coordination, and load-balancing layer,
- semantic object seeking is integrated into graph-level decision making rather than isolated local frontier selection.

## 2. Final System Definition

The target system should combine five capabilities:

1. Multi-agent shared mapping.
2. Zero-shot semantic object navigation.
3. MR-DTG-based global exploration and coordination.
4. Graph-level load balancing across robots.
5. Local geometric execution using occupancy-derived planners.

This is not a pure topological navigation system.

It is a hybrid system:

- occupancy map for geometry,
- semantic fusion for target likelihood,
- MR-DTG for planning and coordination,
- local path planning for execution.

## 3. Core Realization Principle

The realization path should follow this rule:

**do not replace the current low-level map and perception stack first; build a graph-driven planning layer on top of it, then gradually migrate decision logic from frontier mode to MR-DTG mode.**

This minimizes regression risk and keeps the existing system usable during transition.

## 4. High-Level Workflow

The complete workflow should be executed in eight stages:

1. Stabilize the current interfaces and runtime contracts.
2. Separate runtime orchestration from planner logic.
3. Preserve and clean the shared map substrate.
4. Add semantic abstractions for zero-shot object navigation.
5. Build MR-DTG from occupancy-derived free space.
6. Add multi-agent task allocation and load balancing on graph level.
7. Connect graph-level goals to local motion execution.
8. Evaluate, compare, and switch the main planner backend.

## 5. Stage 1: Stabilize Interfaces

### Objective

Freeze the current system boundaries before changing algorithms.

### Tasks

- Fix and document ROS topic contracts.
- Fix and document message formats.
- Fix the config schema for agents, sensors, and planner settings.
- Fix dataset path conventions and runtime assumptions.
- Define the minimal external API between:
  - Habitat runtime,
  - ROS bridge,
  - planner,
  - semantic modules.

### Output

A stable contract document for:

- sensor topics,
- action topics,
- semantic score topics,
- object-cloud topics,
- planner output topics,
- episode/result topics.

### Why this matters

If runtime interfaces change during planner refactor, failures become hard to localize.

## 6. Stage 2: Decouple Runtime Orchestration

### Objective

Split the current monolithic runtime flow into reusable modules.

### Required Refactor

The current evaluation/runtime path should be decomposed into:

1. `habitat_runner`
2. `ros_bridge`
3. `perception_pipeline`
4. `planner_adapter`
5. `evaluation_logger`

### Responsibilities

`habitat_runner`

- creates environments,
- steps simulation,
- resets episodes,
- exposes observations.

`ros_bridge`

- publishes sensor observations,
- receives planner actions,
- handles agent namespaces,
- manages synchronization.

`perception_pipeline`

- runs detector / segmentor / ITM logic,
- produces object point clouds,
- publishes semantic evidence.

`planner_adapter`

- translates between runtime messages and planner-facing state.

`evaluation_logger`

- records metrics,
- saves videos,
- records success/failure metadata.

### Output

A runtime that no longer mixes:

- simulator control,
- semantic inference,
- planner communication,
- and evaluation bookkeeping in one file.

## 7. Stage 3: Preserve and Clean the Shared Map Substrate

### Objective

Keep occupancy-grid mapping as the authoritative geometric substrate.

### What must remain

- occupancy updates,
- inflated occupancy,
- ESDF generation,
- object map,
- value map,
- local updated region bounds,
- geometric visualization.

### Required Cleanup

- clearly separate read-only map queries from write/update operations,
- define a map snapshot interface for graph construction,
- avoid direct planner-side dependence on internal map buffers where possible.

### Output

A clean shared world model that supports:

- local motion planning,
- semantic fusion,
- graph extraction,
- multi-agent coordination.

## 8. Stage 4: Build Zero-Shot Semantic Navigation Abstraction

### Objective

Lift semantic object navigation from local utility functions into a planner-facing abstraction.

### Current problem

The zero-shot object navigation logic is currently scattered across:

- detection,
- segmentation,
- ITM scoring,
- object cloud generation,
- target selection heuristics.

### Required abstraction

Introduce a target belief representation for each candidate object class or object hypothesis.

Each target hypothesis should include:

- target label,
- object evidence cloud,
- semantic confidence,
- ITM score,
- spatial support region,
- freshness timestamp,
- robot-specific visibility state,
- graph-node association.

### Why this matters

MR-DTG planning should reason over semantic targets at graph level, not only at per-frame or per-frontier level.

### Output

A zero-shot target model that can be attached to map regions and graph nodes.

## 9. Stage 5: Construct MR-DTG from Occupancy-Derived Space

### Objective

Introduce the main MR-DTG layer without discarding occupancy mapping.

### Inputs

The MR-DTG builder should read from:

- free space,
- unknown boundaries,
- occupied obstacles,
- ESDF,
- semantic value map,
- object map,
- local map update region.

### MR-DTG graph contents

Each graph node should carry:

- node id,
- representative pose,
- local region descriptor,
- adjacency list,
- traversability score,
- exploration gain,
- semantic gain,
- object-target relevance,
- ownership state,
- last update time.

Each edge should carry:

- edge id,
- connected nodes,
- traversal cost,
- safety cost,
- estimated time-to-traverse,
- validity state.

### Required implementation strategy

The graph builder should be incremental, not global-only.

Recommended approach:

1. detect changed local map region,
2. update only affected nodes and edges,
3. preserve stable graph structure elsewhere,
4. publish graph diffs to the coordinator.

### Output

An MR-DTG graph that represents the explored environment structure more compactly than frontier lists.

## 10. Stage 6: Create a Unified Candidate Interface

### Objective

Make the planner independent from frontier-specific data structures.

### Candidate types

The planner should consume a common exploration candidate interface that can represent:

- frontier candidate,
- MR-DTG node,
- MR-DTG region,
- object-approach waypoint,
- suspicious semantic hotspot.

### Minimal fields

- `id`
- `type`
- `pose`
- `region_id`
- `travel_cost`
- `exploration_gain`
- `semantic_gain`
- `target_relevance`
- `assigned_robot`
- `status`

### Why this matters

This is the main transition mechanism.

During migration:

- frontier candidates can still be produced,
- MR-DTG candidates can be added,
- both can be compared under one planner interface.

### Output

A planner input model that is backend-agnostic.

## 11. Stage 7: Add Multi-Agent Graph-Level Task Allocation

### Objective

Replace loosely independent exploration with coordinated load-balanced assignment.

### Required coordinator

Introduce a `graph_allocator` or equivalent module responsible for:

- assigning robots to nodes or regions,
- balancing travel cost,
- balancing expected exploration workload,
- avoiding duplicated coverage,
- reallocating tasks when graph structure changes,
- considering semantic opportunity when a target is likely in a region.

### Allocation factors

The cost function should consider:

- distance from robot to assigned node,
- expected information gain,
- local unknown-space volume,
- target-object likelihood,
- current robot workload,
- graph congestion or overlap risk,
- communication or synchronization overhead.

### Output

A graph-level assignment result for each robot:

- current target node,
- fallback node,
- assigned subregion,
- target-object priority if applicable.

## 12. Stage 8: Connect MR-DTG Planning to Local Execution

### Objective

Use graph-level planning globally and occupancy-based planning locally.

### Execution logic

1. MR-DTG planner assigns the next graph goal.
2. Local connector computes a feasible route from robot pose to node entry or representative pose.
3. If needed, kinodynamic planner refines the route.
4. The robot executes motion while local map and graph continue updating.

### Required local planner role

The local planner should provide:

- reachability check,
- travel cost estimate,
- safe path generation,
- replanning on local failure.

### Important constraint

`Astar2D` should remain a local geometric tool, not the dominant global decision engine.

## 13. Stage 9: Merge Zero-Shot Object Seeking With MR-DTG Exploration

### Objective

Unify zero-shot object navigation and multi-robot exploration into one planner.

### Planning modes

The planner should support three coordinated modes:

1. Exploration mode
2. Object-seeking mode
3. Hybrid mode

### Exploration mode

Used when no strong semantic evidence exists.

Behavior:

- allocate graph regions,
- maximize new-space coverage,
- maintain load balance.

### Object-seeking mode

Used when target evidence is strong.

Behavior:

- rank graph nodes by target relevance,
- allocate nearest or best-positioned robot,
- preserve coverage with remaining robots.

### Hybrid mode

Used when the target is uncertain but semantic evidence exists.

Behavior:

- keep graph exploration active,
- bias node selection toward target-related regions,
- allow one robot to exploit target evidence while others maintain exploration.

### Output

A unified policy that treats object navigation and exploration as coordinated graph-level decisions.

## 14. Stage 10: Make Multi-Agent Support Dynamic

### Objective

Remove fixed two-agent assumptions.

### Required changes

- make robot count configurable,
- dynamically create subscriptions and publishers,
- generalize ownership and assignment tables,
- generalize progress and result reporting,
- generalize robot-state registration.

### Output

A planner and runtime path that scales beyond a fixed two-robot setup.

## 15. Stage 11: Validation and Benchmark Workflow

### Objective

Measure whether MR-DTG mode is actually better than the current frontier-first mode.

### Required evaluation tracks

1. Single-robot baseline.
2. Current multi-robot frontier baseline.
3. MR-DTG without semantic bias.
4. MR-DTG with semantic zero-shot bias.
5. MR-DTG with load balancing enabled.

### Metrics

- success rate,
- SPL or path-efficiency metric,
- target-found time,
- explored-area rate,
- duplicated exploration ratio,
- assignment imbalance,
- communication payload or structural update volume,
- replanning frequency,
- average travel cost per robot.

### Output

A benchmark matrix showing whether MR-DTG improves:

- coordination,
- exploration efficiency,
- load balance,
- and zero-shot object acquisition.

## 16. Recommended Implementation Order

The practical implementation order should be:

1. Document and freeze interfaces.
2. Split runtime orchestration modules.
3. Clean the shared map access layer.
4. Introduce semantic target belief structures.
5. Implement MR-DTG builder.
6. Implement unified candidate interface.
7. Implement graph allocator.
8. Integrate graph planner with local execution.
9. Add hybrid exploration plus object-seeking policy.
10. Generalize multi-agent count.
11. Benchmark and switch default backend.

## 17. Concrete Module Plan

Suggested modules to add or refactor:

### New modules

- `mr_dtg_builder`
- `mr_dtg_graph`
- `graph_allocator`
- `graph_planner`
- `semantic_target_model`
- `planner_adapter`

### Existing modules to preserve

- occupancy and ESDF mapping
- object map
- value map
- local path planner
- Habitat ROS bridge

### Existing modules to demote from core role

- frontier-only exploration selection
- frontier claiming as the primary coordination mechanism
- monolithic runtime evaluator

## 18. Minimum Definition of Done

The refactor can be considered structurally complete when:

- occupancy map remains the geometric authority,
- MR-DTG is incrementally built from shared explored space,
- graph-level planning is the main multi-robot coordination layer,
- semantic target evidence is attached to graph structure,
- task allocation is load-balanced across robots,
- local motion execution is occupancy-based,
- frontier logic is optional or auxiliary instead of primary.

## 19. Summary

The realization workflow for multi-agent zero-shot object navigation should not begin by replacing the entire current system.

It should proceed by layering MR-DTG on top of the current occupancy-map and semantic-fusion base, then migrating planning authority upward from local frontier logic to graph-level coordination.

In short, the build path is:

1. keep geometric mapping,
2. clean runtime boundaries,
3. elevate semantic target reasoning,
4. build MR-DTG,
5. allocate multi-robot work on the graph,
6. execute locally with occupancy-aware motion planning,
7. validate against the frontier baseline.

That is the safest and most technically coherent path to realize multi-agent zero-shot object navigation in this stack.
