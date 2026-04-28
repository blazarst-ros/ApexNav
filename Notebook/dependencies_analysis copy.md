# Dependencies and MR-DTG Refactoring Specification

## 1. Scope

This document defines the shared dependencies, coupling points, and refactoring prerequisites for `ApexNav`, with the specific target of evolving the current multi-robot exploration stack toward an **MR-DTG (Multi-Robot Dynamic Topology Graph)** mode.

The goal is not to discard the current occupancy-grid-based mapping pipeline.

The intended direction is:

- keep occupancy maps as the geometric perception and visualization substrate,
- keep semantic and object fusion on top of that substrate,
- introduce MR-DTG as the high-level planning and coordination representation,
- use the topological graph for multi-robot exploration planning and load balancing.

This means the refactor should be understood as an **occupancy-map-backed MR-DTG planning architecture**, not as a pure topological navigation system.

## 2. Problem Statement

The original multi-robot navigation implemented in ApexNav is built around independent exploration on occupancy grid maps.

This design has practical limitations for multi-robot exploration:

- limited communication efficiency,
- weak task allocation structure,
- duplicated exploration effort between robots,
- delayed coordination when map updates are merged,
- poor load balancing when robots operate in spatially asymmetric environments,
- frontier selection that is locally effective but not globally coordinated.

Your proposed solution is to use **MR-DTG** for environmental perception and coordination, while preserving occupancy-based mapping as the low-level spatial representation.

In this design:

- occupancy maps remain the base layer for sensing, collision checking, and visualization,
- topological graph structures are constructed from the shared explored space,
- exploration planning is performed over the graph rather than directly over frontier cells,
- multi-robot load balancing is optimized at graph level.

## 3. Target System Interpretation

To avoid ambiguity, the target architecture should be interpreted as follows.

### 3.1 What Should Remain Occupancy-Based

The following should remain occupancy-grid or grid-derived:

- depth ingestion,
- free/unknown/occupied map updates,
- inflated occupancy map,
- ESDF construction,
- collision checking,
- object evidence projection,
- semantic value accumulation,
- RViz and map visualization.

### 3.2 What Should Become MR-DTG-Based

The following should move to MR-DTG-mode logic:

- exploration-space abstraction,
- robot coordination,
- region/node assignment,
- global route scheduling,
- load balancing,
- communication-efficient frontier or region exchange,
- dynamic replanning after map growth.

### 3.3 Key Clarification

The target is **not**:

- replacing occupancy mapping with a pure topological map,
- removing the current geometric map substrate,
- treating topological nodes as the only environmental representation.

The target **is**:

- using occupancy maps to support perception and visualization,
- deriving a dynamic topology graph from explored free space,
- planning and allocating multi-robot exploration tasks over that graph.

## 4. Existing Architecture Summary

The current repository has three tightly coupled layers.

### 4.1 Python Habitat Runtime Layer

The Python runtime is centered on `habitat_evaluation.py`, which currently performs:

- Habitat environment setup,
- ROS orchestration,
- simulator stepping,
- observation publication,
- semantic perception calls,
- object-point-cloud generation,
- optional LLM answer loading,
- evaluation and logging.

This file is effectively a runtime orchestrator rather than just an evaluator.

### 4.2 ROS/C++ Mapping and Planning Layer

The core planner stack is under `src/planner` and includes:

- `plan_env`
- `path_searching`
- `exploration_manager`
- `trajectory_manager`
- `utils/lkh_mtsp_solver`
- `utils/vis_utils`

The current planner is built around a shared 2D map state and a frontier-first decision model.

### 4.3 Vision-Language and Semantic Fusion Layer

The current semantic stack includes:

- GroundingDINO,
- MobileSAM,
- YOLO-based detection,
- BLIP2 ITM scoring,
- optional OpenAI / Ollama support.

This layer generates semantic evidence that is fused into the shared map and then consumed by the planner.

## 5. Shared Technical Dependencies

The refactor must preserve the current dependency surfaces before introducing MR-DTG.

### 5.1 Runtime and Environment Dependencies

Current environment assumptions include:

- Ubuntu 20.04,
- ROS Noetic,
- Python 3.9,
- Habitat-Sim 0.3.1,
- Habitat-Lab 0.3.1,
- Catkin workspace build flow,
- Conda-based Python environment.

Important Python/runtime dependencies include:

- `numpy == 1.23.5`
- `numba == 0.60.0`
- `rospkg == 1.5.1`
- `open3d == 0.18.0`
- `transformers == 4.43.2`
- `ultralytics == 8.3.39`
- `fastapi == 0.111.1`
- `uvicorn == 0.30.3`
- `openai >= 1.58.1`
- `ollama`
- `mobile_sam`
- `groundingdino`
- local package `habitat2ros`

Important system/build dependencies include:

- `PCL`
- `Eigen3`
- `OpenCV`
- `cv_bridge`
- `message_filters`
- `ompl`
- `OsqpEigen`
- `libarmadillo-dev`
- `libompl-dev`

### 5.2 ROS Topic Contract Dependencies

The system relies on a fixed simulator-to-planner ROS topic contract.

Representative topics include:

- `/habitat/agent_X/camera_depth`
- `/habitat/agent_X/camera_rgb`
- `/habitat/agent_X/odom`
- `/habitat/agent_X/sensor_pose`
- `/habitat/plan_action_agent_X`
- `/blip2/agent_X/cosine_score`
- `/detector/agent_X/clouds_with_scores`
- `/ros/state_all`
- `/ros/expl_state`
- `/ros/expl_result`

These contracts should remain stable during the MR-DTG migration.

### 5.3 Shared Map-State Dependencies

The entire current stack depends on a shared map state centered on `SDFMap2D`.

This state currently supports:

- occupancy,
- inflated occupancy,
- ESDF,
- object map,
- value map,
- local update bounds.

This is an important advantage.

MR-DTG does not need to replace this layer. It should be constructed **on top of it**.

### 5.4 Multi-Agent Shared-State Dependencies

The current system uses:

- per-agent sensor state,
- shared global occupancy/object/value map,
- frontier-claim logic,
- fixed two-agent assumptions.

This design already provides a merged-world representation, but it does not provide a graph-native coordination layer.

## 6. Current Limitations Relative to MR-DTG

### 6.1 Frontier-First Decision Model

The current planner is built around frontier exploration.

`FrontierMap2D` defines active and dormant frontier clusters, and `ExplorationManager` selects targets from those structures.

This creates a mismatch with MR-DTG because:

- the planner reasons over local frontier sets,
- coordination is target-claim-based rather than graph-structured,
- global exploration structure is weak,
- balancing between robots is heuristic rather than topology-aware.

### 6.2 Independent Exploration Bias

Even though robots share map information, the active logic still trends toward independent exploration with local deconfliction.

This limits:

- global division of labor,
- coordinated path decomposition,
- graph-level task scheduling,
- efficient communication of only the necessary structural updates.

### 6.3 Occupancy Map Exists, But Graph Layer Does Not

The current code already has the correct geometric substrate for MR-DTG:

- explored free space,
- obstacles,
- semantics,
- objects,
- local update windows.

What is missing is an explicit dynamic topology graph layer that converts this shared map into:

- graph nodes,
- graph edges,
- graph regions,
- robot assignments,
- dynamic load metrics.

### 6.4 Multi-Robot Coordination is Not Load-Balancing-Oriented

The current coordination logic is primarily based on frontier claiming and per-agent planning.

This is weaker than MR-DTG-style coordination because it does not explicitly optimize:

- workload distribution,
- travel cost balance,
- regional decomposition,
- graph-subtree ownership,
- communication-efficient structural synchronization.

## 7. MR-DTG Refactor Objectives

The MR-DTG mode for ApexNav should satisfy the following objectives.

### 7.1 Preserve Occupancy-Map Perception

The occupancy-grid pipeline should remain the authoritative source for:

- sensor fusion,
- collision information,
- ESDF updates,
- geometric free-space extraction,
- semantic and object overlays,
- visualization.

### 7.2 Build Dynamic Topology Graph from Occupancy-Derived Free Space

The system should construct an MR-DTG representation from the current map state.

The graph should contain:

- graph nodes representing key structural locations or regions,
- graph edges representing traversable adjacency,
- dynamic updates when explored space changes,
- topological connectivity between newly discovered and existing regions.

### 7.3 Use MR-DTG for Multi-Robot Planning

Exploration planning should move from direct frontier ranking to graph-level decision making.

The graph planner should support:

- assigning robots to graph regions or nodes,
- estimating marginal exploration gain,
- balancing travel effort across robots,
- reducing duplicated exploration,
- replanning after dynamic map expansion.

### 7.4 Support Communication-Efficient Coordination

MR-DTG should improve communication efficiency by reducing the need to exchange dense frontier-level state as the main coordination signal.

Preferred exchange units should become:

- graph node updates,
- graph edge changes,
- region ownership,
- summarized semantic or exploration utility.

## 8. Required Architectural Shift

The correct shift is not from occupancy map to topological map.

The correct shift is from:

- occupancy-map-driven independent frontier exploration,

to:

- occupancy-map-backed MR-DTG planning and coordination.

This means the architectural layering should become:

1. geometric map layer,
2. semantic/object fusion layer,
3. MR-DTG extraction layer,
4. multi-robot task-allocation layer,
5. local motion layer.

## 9. Refactoring Prerequisites

### 9.1 Freeze the Existing Map and ROS Interfaces

Before introducing MR-DTG, preserve:

- message types,
- topic contracts,
- config schema,
- Habitat observation assumptions,
- object cloud publication format,
- semantic score publication format.

Reason:

The MR-DTG refactor should be isolated from transport-layer regressions.

### 9.2 Keep `SDFMap2D` as the Base World Model

`SDFMap2D` should remain the base substrate for:

- occupancy,
- inflation,
- ESDF,
- local update bounds.

Its current role should be preserved.

Reason:

MR-DTG should be derived from occupancy structure rather than replacing it.

### 9.3 Introduce an Explicit MR-DTG Construction Module

A new graph construction module should be added between the map layer and planner layer.

This module should:

- read occupancy and free-space structure,
- derive dynamic topological nodes,
- derive graph edges,
- maintain incremental graph updates,
- expose graph metadata to planners.

Recommended output fields:

- node id,
- node pose or representative location,
- local region coverage score,
- semantic score,
- object relevance score,
- adjacency list,
- traversal cost,
- robot ownership state,
- update timestamp.

### 9.4 Replace Frontier as the Primary Exploration Primitive

The planner should no longer be centered directly on frontier clusters.

Instead:

- frontier information may remain as one source of local evidence,
- MR-DTG nodes or regions should become the primary planning units,
- frontier clusters may be attached to graph nodes as local expansion cues.

Reason:

This preserves useful frontier information without keeping frontier logic as the dominant planning abstraction.

### 9.5 Introduce Graph-Level Task Allocation and Load Balancing

A new coordination module is required for:

- graph-region assignment,
- path-cost-aware load balancing,
- reallocation when one robot finishes early,
- conflict avoidance in shared graph neighborhoods,
- balancing semantic opportunity against traversal cost.

This module is the core functional difference between the current design and the MR-DTG target mode.

### 9.6 Separate Global Planning from Local Motion Execution

The new architecture should divide planning into:

- graph-level global exploration planning,
- local geometric path connection,
- optional kinodynamic or trajectory execution.

`Astar2D` should remain useful as a local feasibility or connector tool, but not as the central global exploration primitive.

### 9.7 Generalize Beyond Two Hardcoded Agents

The current two-agent assumption should be removed.

The following should become dynamic:

- number of robots,
- sensor subscription generation,
- graph ownership tables,
- assignment results,
- task-progress structures,
- robot-specific semantic output channels.

Reason:

MR-DTG is inherently a scalable coordination abstraction.

### 9.8 Split Runtime Orchestration From Evaluation Logic

`habitat_evaluation.py` should be separated into narrower components.

Recommended decomposition:

- Habitat runtime runner,
- ROS bridge manager,
- perception pipeline manager,
- planner integration manager,
- evaluator and logger.

Reason:

MR-DTG integration will otherwise remain entangled with evaluation and simulator-control logic.

## 10. Recommended MR-DTG-Oriented Architecture

A practical target architecture is:

### 10.1 Layer 1: Geometric Perception Layer

Responsibilities:

- occupancy map updates,
- free/unknown/occupied maintenance,
- inflated occupancy,
- ESDF updates,
- low-level visualization.

Primary owner:

- `SDFMap2D` and `MapROS`

### 10.2 Layer 2: Semantic and Object Fusion Layer

Responsibilities:

- object evidence integration,
- semantic value updates,
- target-centric semantic fusion,
- region-level semantic accumulation.

Primary owners:

- `ObjectMap2D`
- `ValueMap`
- VLM/ITM utilities

### 10.3 Layer 3: MR-DTG Extraction Layer

Responsibilities:

- derive topological graph from explored space,
- update graph incrementally as occupancy changes,
- associate graph nodes with local free-space regions,
- attach semantic and object summaries to graph nodes.

This is the missing layer in the current codebase.

### 10.4 Layer 4: Multi-Robot Coordination Layer

Responsibilities:

- assign graph nodes or regions to robots,
- optimize load balancing,
- minimize overlap,
- support coordination-aware replanning,
- exchange compact graph updates.

### 10.5 Layer 5: Local Motion and Execution Layer

Responsibilities:

- connect robot pose to assigned graph node or entry point,
- run local path search,
- optionally run kinodynamic planning and trajectory optimization.

## 11. Transitional Migration Strategy

A safe migration path is:

### Phase 1: Preserve the Existing Geometric Mapping Stack

Do not rewrite:

- occupancy updates,
- ESDF logic,
- object fusion,
- value-map logic,
- ROS sensing contracts.

### Phase 2: Introduce MR-DTG in Parallel

Add a graph module that reads the existing map state and produces topological structures without removing frontier logic yet.

### Phase 3: Wrap Exploration Targets Behind a Unified Interface

Create a generic exploration-target abstraction that can represent:

- frontier cluster,
- MR-DTG node,
- MR-DTG region,
- object-approach waypoint.

This allows side-by-side evaluation.

### Phase 4: Move Multi-Robot Assignment to Graph Level

Replace frontier-claim-based coordination with:

- graph ownership,
- region allocation,
- load-aware reassignment.

### Phase 5: Keep Frontiers Only as Auxiliary Local Signals

Once MR-DTG planning is stable:

- frontiers should be treated as local exploration evidence,
- not as the main global coordination representation.

## 12. Main Technical Risks

### 12.1 Over-Replacing the Occupancy Layer

A major conceptual risk is attempting to replace occupancy maps with pure topology.

That would be a mistake for this repo.

ApexNav already depends heavily on occupancy-derived sensing, ESDF, and semantic fusion.

MR-DTG should sit above that substrate.

### 12.2 Graph Update Cost

If the topology graph is rebuilt globally every cycle, runtime cost may become unacceptable.

Mitigation:

- use local update bounds,
- support incremental node/edge updates,
- separate slow graph maintenance from fast local control.

### 12.3 Weak Semantic Integration at Graph Level

If semantics remain only cell-based, MR-DTG planning quality will be limited.

Mitigation:

- aggregate semantic and object evidence per graph node or region,
- maintain freshness and confidence of node-level semantic estimates.

### 12.4 Migration Risk From Frontier Logic

Removing frontier logic too early will make benchmarking difficult.

Mitigation:

- keep frontier logic during transition,
- compare current frontier baseline against MR-DTG mode.

## 13. Minimum Acceptance Criteria for MR-DTG Mode

Before considering the refactor base complete, the system should satisfy:

- occupancy map remains the authoritative geometric representation,
- MR-DTG is incrementally derived from explored space,
- global exploration planning operates on graph structure rather than direct frontier ranking,
- multi-robot task allocation is graph-level and load-aware,
- robot coordination no longer relies mainly on independent frontier claiming,
- semantic/object fusion can be attached to graph nodes or regions,
- the number of robots is configurable rather than fixed at two.

## 14. Conclusion

The correct evolution path for ApexNav is not a generic switch from occupancy maps to topological maps.

It is a more specific architectural shift:

- keep occupancy maps for sensing, geometry, ESDF, and visualization,
- derive an MR-DTG representation from the explored occupancy structure,
- use MR-DTG as the main planning and coordination layer for multi-robot exploration,
- improve communication efficiency and load balancing through graph-level reasoning.

In short:

**ApexNav should move from independent occupancy-grid frontier exploration to occupancy-map-backed MR-DTG exploration planning.**

That is the right interpretation of the target mode, and the refactor should be scoped accordingly.
