
# Replace Per-Step MTSP With Voronoi-Region Task
  Allocation

  ## Summary

  Use a Voronoi-region allocator instead of the current
  global MTSP-style assignment. This is the better fit
  for your two problems:

  - Current allocator is expensive because it recomputes
    global routes and calls A* many times every planning
    step.
  - Current assignment can send an agent to far
    frontiers because global route balancing and
    semantic priority can overpower locality.
  - Voronoi allocation makes each agent responsible for
    nearby frontier/object regions first, then uses
    local policy inside that region.

  This should improve step-time and make adjacent-area
  behavior more natural.

  ## Key Changes

  ### Allocation Strategy

  - Replace planMultiAgentAssignments() with
    planVoronoiAssignments().
  - Partition candidate tasks by nearest active agent
    using Euclidean distance first.
  - Add a path-cost refinement only for each agent’s top
    local candidates, not for every task-agent pair.
  - Default candidate priority inside each region:
      - strict object task
      - suspicious object task
      - high-semantic frontier
      - nearest reachable frontier
      - dormant fallback
  - Keep previous assignment if still valid and not much
    worse than the best new candidate. This hysteresis
    prevents task flipping every step.

  ### Locality And Adjacent Area Behavior

  - Add a locality gate:
      - agent can only take tasks in its Voronoi region
        by default.
      - allow stealing boundary tasks only if its
        estimated cost is at least 30% better than the
        owner’s cost.
  - Add a max-local-radius rule:
      - ignore far frontiers while nearer reachable
        frontiers exist in the same region.
      - suggested default: local_frontier_radius = 4.0m.
  - Add region fallback:
      - if an agent’s region has no reachable tasks, it
        can borrow from the nearest neighboring region.
      - borrowing must choose the nearest reachable
        task, not highest global semantic score.

  ### Runtime Cost Reduction

  - Do not run full A* for all (agent, task) pairs.
  - Use this staged cost model:
      - Stage 1: Euclidean distance for Voronoi
        partition and candidate sorting.
      - Stage 2: A* only for top K=5 candidates per
        agent.
      - Stage 3: exact path stored only for the selected
        task.
  - Cache task signatures:
      - task position rounded to grid cell
      - task type
      - agent id
      - map/frontier update counter
  - Reuse cached A* costs until frontier/object map
    changes.

  ### Interfaces And Data

  - Keep existing mtsp_tours_, mtsp_assigned_task_pos_,
    mtsp_assigned_task_type_, and RViz route fields for
    compatibility.
  - Change their meaning:
      - mtsp_tours_[i] becomes voronoi_route_[i]
        logically, but can keep the same field name to
        avoid broad rewiring.
      - Each route contains agent pose plus ordered
        local candidates.
  - Add debug fields/topic content:
      - assigned task owner
      - region size
      - selected candidate rank
      - Euclidean cost
      - A* cost
      - whether task was borrowed from another region

  ## Implementation Changes

  - In ExplorationManager:
      - Replace global min-max task loop with Voronoi
        partitioning.
      - Add selectVoronoiTaskForAgent().
      - Add top-K candidate A* refinement.
      - Add assignment hysteresis using previous
        assigned task.
      - Keep old single-agent fallback if only one
        active agent exists.
  - In simulation FSM:
      - Call planVoronoiAssignments() once per planning
        cycle before per-agent planning.
      - Keep consumeAssignedTask() behavior.
  - In RViz:
      - Keep per-agent route visualization.
      - Add optional boundary-region colors by drawing
        each agent’s assigned frontier/object candidates
        in that agent color.
  - Keep /solve_tsp and /solve_mtsp services available,
    but do not call them during normal simulation
    allocation.

  ## Test Plan

  - Performance:
      - Compare average planning time before/after.
      - Expected: fewer A* calls and lower per-step
        latency.
  - Locality:
      - Put 3 agents near 3 adjacent frontier clusters.
      - Expected: each agent receives nearby cluster,
        not far global frontier.
  - Boundary behavior:
        neighboring region.
  - Semantic object case:
      - Strict target appears near one agent.
      - Expected: nearest reachable agent takes it;
        others continue local frontiers.
  - Regression:
        local path, and traveled path.

  ## Assumptions

  - Implement Voronoi-region allocation for simulation
    mode first.
  - Optimize for adjacent-area locality and fast
    replanning over perfect global tour optimality.
  - Keep existing message/API fields where possible.
  - Do not remove LKH/TSP solver packages; they remain
    fallback tools, not the default allocator.
  - Default settings:
      - top_k_refine = 5
      - local_frontier_radius = 4.0m
      - boundary_steal_ratio = 0.70
      - assignment hysteresis keeps prior task unless
        new task is at least 20% better