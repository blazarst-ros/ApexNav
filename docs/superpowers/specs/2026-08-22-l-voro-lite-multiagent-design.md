# l-Voro Lite Multi-Agent Design

## Goal

Run the existing two-agent `l-Voro` navigation logic with the Lite-ApexNav
runtime: YOLOE on port 12184, CLIPITM on port 12182, and bundled offline
semantic answers. Preserve all planner, ROS, message, reach-claim, Stage-1,
and non-protection behavior.

## Architecture

`habitat_evaluation.py` and `habitat_manual_control_multiagent.py` retain their
two-agent loops and ROS interfaces. Their existing helper imports become a
compatibility seam: `get_object()` is backed by YOLOE segmentation and
`get_itm_message_cosine()` by CLIPITM, while keeping return shapes and the
legacy `/blip2/agent_X/cosine_score` topic contract.

The existing ignored Habitat-Lab v0.3.1 checkout supplies
`MultiAgentSim-v0`. Its five-file patch is stored in the ApexNav repository as
a reproducible patch pinned to base commit
`142616776544f918c19e7f0392b65cc8cc69fa13`; a setup script verifies and
applies it, and a repository-owned runner puts that checkout before the Lite
environment's Habitat installation on `PYTHONPATH`.

## Constraints

- Do not merge or copy the Lite single-agent planner/C++ implementation.
- Do not modify `src/planner/exploration_manager/**`, MapROS/ObjectMap source,
  `MultipleMasksWithConfidence.msg`, the namespaced Habitat ROS publisher, or
  the per-agent point-cloud helper.
- Keep target label index 0 and publish the complete `class_names` dictionary.
- Keep `/blip2/agent_X/cosine_score` and
  `/detector/agent_X/clouds_with_scores` unchanged.
- No runtime request may use GroundingDINO, YOLOv7, MobileSAM, BLIP2,
  DeepSeek, Ollama, or an external LLM API.
- HM3D-v2 is the end-to-end acceptance dataset; HM3D-v1 and MP3D configs must
  remain structurally compatible.

## Acceptance

The Lite Python environment must load the pinned local Habitat patch, register
`MultiAgentSim-v0`, compose two agents with namespaced sensor UUIDs, and run
both automatic and manual two-agent entrypoints using only CLIPITM and YOLOE.
All existing multi-agent/non-protection policy tests must continue to pass.
