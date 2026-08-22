# l-Voro Lite Multi-Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve l-Voro's two-agent navigation while replacing its runtime perception and semantic-answer stack with Lite-ApexNav YOLOE, CLIPITM, and offline answers.

**Architecture:** Keep all current multi-agent entry loops and ROS/C++ contracts. Replace only the Python VLM helper implementations and detector configuration, then make the existing Habitat-Lab `MultiAgentSim-v0` patch reproducible through a pinned patch and runner.

**Tech Stack:** Python 3.9, ROS Noetic, Habitat-Lab/Habitat-Sim 0.3.1, Hydra/OmegaConf, Flask, YOLOE/Ultralytics, CLIP, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-l-voro-lite-multiagent-design.md`

## Global Constraints

- Keep the l-Voro C++ planner, message definitions, namespaced ROS publisher, point-cloud helper, reach-claim, Stage-1, and non-protection behavior unchanged.
- Preserve `/blip2/agent_X/cosine_score`, `/detector/agent_X/clouds_with_scores`, target label index 0, and `MultipleMasksWithConfidence.class_names`.
- Runtime services are exactly CLIPITM port 12182 and YOLOE port 12184; no GroundingDINO, YOLOv7, MobileSAM, BLIP2, DeepSeek, Ollama, or external LLM request.
- Pin Habitat-Lab patch base to `142616776544f918c19e7f0392b65cc8cc69fa13`.
- Use TDD for behavioral code and commit each task independently.

---

### Task 1: Lite VLM compatibility seam

**Files:**
- Create: `vlm/detector/yoloe.py`, `vlm/itm/clipitm.py`, `vlm/label_utils.py`, `basic_utils/path_utils.py`
- Modify: `vlm/detector/detections.py`, `vlm/server_wrapper.py`, `vlm/utils/get_itm_message.py`, `vlm/utils/get_object_utils.py`, `pyproject.toml`
- Test: `tests/test_lite_vlm_contract.py`

**Interfaces:**
- `get_object(right_label, img, cfg, similar_answer, return_stats=False)` retains the four-value default return and may return a fifth timing dictionary only when requested.
- `get_object_class_names(right_label, similar_answer)` returns target first and de-duplicated confusion labels.
- `get_itm_message_cosine(rgb_image, label, room, return_stats=False)` returns float by default.

- [ ] Write contract tests that fail because YOLOE/CLIP modules and behavior are absent.
- [ ] Run the focused tests and capture the expected RED result.
- [ ] Port the Lite clients/server wrapper and adapt `get_object_utils` without importing legacy model clients.
- [ ] Update dependency declarations for YOLOE/CLIP and remove runtime-only heavy-model dependencies.
- [ ] Run focused and existing detector tests, then commit.

### Task 2: Offline answers and two-agent entry/config integration

**Files:**
- Modify: `llm/answer.py`, `llm/answer_reader/answer_reader.py`, `config/habitat_eval_hm3dv1.yaml`, `config/habitat_eval_hm3dv2.yaml`, `config/habitat_eval_mp3d.yaml`, `habitat_evaluation.py`, `habitat_manual_control_multiagent.py`
- Test: `tests/test_lite_offline_multiagent_contract.py`

**Interfaces:**
- `read_answer(...)` continues to return `(similar_labels, room, fusion_threshold)` and never calls a client in offline mode.
- All three configs retain `num_agents: 2`, agent topology, heights, multiagent scheduling, dataset paths, and success semantics; only detector keys and LLM mode change.
- Existing `/blip2/agent_X` and detector cloud topics stay unchanged.

- [ ] Write tests for cached/unknown offline answers, config invariants, imports, and unchanged ROS topic contracts; verify RED.
- [ ] Implement deterministic offline answer reading with neutral unknown-label defaults.
- [ ] Replace detector config sections with `detector.yoloe` and select `llm_client: offline`.
- [ ] Adapt automatic/manual entrypoints only where required by the helper/config seam.
- [ ] Run focused tests plus multi-agent/non-protection regression tests, then commit.

### Task 3: Reproducible MultiAgentSim and operating guide

**Files:**
- Create: `patches/habitat-lab-v0.3.1-multi-agent.patch`, `scripts/setup_lite_multiagent_habitat.sh`, `scripts/run-lite-multiagent.sh`, `liteOp.txt`
- Modify: `.gitignore` only if needed for generated patch/setup artifacts
- Test: `tests/test_lite_multiagent_runtime_setup.py`

**Interfaces:**
- Setup verifies Habitat-Lab base commit before applying the patch and is idempotent.
- Runner prepends `/home/blazarst/ApexNav/habitat-lab/habitat-lab` to `PYTHONPATH`, uses the existing Lite environment/model caches, sources ROS/devel, and accepts arbitrary Python arguments.

- [ ] Write static/runtime setup tests and verify RED.
- [ ] Generate a tracked patch containing the four modified Habitat-Lab files and `multi_agent_sim.py`, pinned to the exact v0.3.1 base.
- [ ] Implement idempotent setup and runner scripts with actionable failures.
- [ ] Document build, two VLM services, planner, automatic evaluation, manual control, and health checks.
- [ ] Verify patch applicability, registration, config composition, and script tests, then commit.

### Task 4: Integrated verification

**Files:**
- Modify only files required to repair integration failures discovered by tests; no scope expansion.

- [ ] Run all Python policy/contract tests.
- [ ] Run Python compile checks for changed modules.
- [ ] Run Habitat registration and two-agent config composition using the Lite Python and local patched Habitat checkout.
- [ ] Run ROS/catkin build verification because Python messages and C++ consumers share the workspace.
- [ ] If live model services are available, run health checks; otherwise report the exact deferred live smoke commands.
- [ ] Commit only genuine integration fixes, if any.
