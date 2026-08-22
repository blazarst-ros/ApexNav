# Task 2 implementation report

## Summary

- `read_answer(...)` now recognizes the configured `offline` client before any
  answer request.  A valid cached legacy answer is still decoded into
  `(similar_labels, room, fusion_threshold)`; an unknown or invalid offline
  label deterministically returns `([], "unknown", 0.4)` without writing
  answer/response files or calling `get_answer(...)`.
- The HM3D-v1, HM3D-v2, and MP3D evaluation configs now select the Lite
  `detector.yoloe` settings and `llm_client: offline`.  Their two-agent
  topology, heights, scheduling, dataset paths, and cooperative success
  semantics are unchanged.
- The automatic and manual multi-agent entrypoints already consumed the Task 1
  helper seam while preserving the per-agent ITM and point-cloud topic names,
  `class_names`, and label-index flow.  They therefore need no behavioral
  edit; the new contract test pins those imports and ROS handoffs.

## TDD evidence

RED command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /media/blazarst/Getea/Lite-ApexNav/conda-env/Lite-apex/bin/python \
  -m unittest discover -s tests -p 'test_lite_offline_multiagent_contract.py' -v
```

Observed expected failures before implementation:

- configs retained `detector.yolo`/`groundingDINO` instead of exactly
  `detector.yoloe`;
- an uncached `offline` label called the patched `get_answer(...)` client.

GREEN / focused verification:

```bash
PYTHONDONTWRITEBYTECODE=1 YOLO_CONFIG_DIR=/tmp/apexnav-yolo-config \
  /media/blazarst/Getea/Lite-ApexNav/conda-env/Lite-apex/bin/python \
  -m unittest discover -s tests -p 'test_lite_offline_multiagent_contract.py' -v
```

Result: 4 tests passed.

Regression verification:

```bash
PYTHONDONTWRITEBYTECODE=1 YOLO_CONFIG_DIR=/tmp/apexnav-yolo-config \
  /media/blazarst/Getea/Lite-ApexNav/conda-env/Lite-apex/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

Result: 14 `unittest` contract tests passed, including Lite VLM and detector
routing.  The source-level existing multi-agent/non-protection regression
functions were also invoked directly (the Lite environment lacks `pytest`):
12 passed.

Compile and diff checks:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /media/blazarst/Getea/Lite-ApexNav/conda-env/Lite-apex/bin/python \
  -m py_compile llm/answer.py llm/answer_reader/answer_reader.py \
  habitat_evaluation.py habitat_manual_control_multiagent.py \
  tests/test_lite_offline_multiagent_contract.py
git diff --check
```

Result: both commands exited successfully.

## Commit

`HEAD` — `feat: configure offline Lite multi-agent evaluation`

## Changed files

- `config/habitat_eval_hm3dv1.yaml`
- `config/habitat_eval_hm3dv2.yaml`
- `config/habitat_eval_mp3d.yaml`
- `llm/answer_reader/answer_reader.py`
- `tests/test_lite_offline_multiagent_contract.py`
- `.superpowers/sdd/2026-08-22-l-voro-lite-multiagent/task-2-implementer-report.md`

## Risks / deferred verification

- No model server, external LLM, ROS node, or Habitat runtime was started.
- The Lite Python environment has no `pytest`, so function-style policy tests
  were run directly rather than through pytest collection.
- `tests/test_two_agent_topology.py` cannot complete in this worktree because
  the pre-existing `RuntimeData/README.md` and
  `RuntimeData/capture_ros_data.sh` paths are absent.  This task does not
  create or modify those runtime/Task 3 assets.
