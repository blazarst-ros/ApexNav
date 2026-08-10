# Exclusive Detector Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route COCO targets exclusively to YOLOv7 and non-COCO targets exclusively to GroundingDINO.

**Architecture:** Add a small pure selector that chooses a detector from `right_label` only. `get_object` passes only labels supported by the selected detector and never calls the other client's HTTP endpoint.

**Tech Stack:** Python 3.8, unittest, existing VLM clients.

## Global Constraints

- COCO target labels use YOLOv7 only.
- Non-COCO target labels use GroundingDINO only.
- Similar-answer labels must not change the detector choice.
- No fallback HTTP request to the other model.

---

### Task 1: Test and implement target-based routing

**Files:**
- Modify: `vlm/utils/get_object_utils.py`
- Create: `tests/test_detector_routing.py`

**Interfaces:**
- Consumes: `right_label: str`, `similar_answer: list[str]`, `COCO_CLASSES`.
- Produces: `(detector_name, detector_labels)` from `_select_detector_labels`.

- [ ] **Step 1: Write the failing tests**

```python
def test_coco_target_uses_yolo_even_when_similar_answer_is_not_coco():
    assert _select_detector_labels("chair", ["furniture"]) == ("yolo", ["chair"])

def test_non_coco_target_uses_grounding_dino_even_when_similar_answer_has_coco_labels():
    assert _select_detector_labels("ottoman", ["chair"]) == ("gdino", ["ottoman", "chair"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_detector_routing -v`

Expected: FAIL because the selector is absent.

- [ ] **Step 3: Implement minimal routing**

```python
if all(label in COCO_CLASSES for label in target_labels):
    return "yolo", [label for label in all_labels if label in COCO_CLASSES]
return "gdino", all_labels
```

- [ ] **Step 4: Run regression tests**

Run: `python3 -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add vlm/utils/get_object_utils.py tests/test_detector_routing.py docs/superpowers/plans/2026-08-10-exclusive-detector-routing.md
git commit -m "fix: route COCO detection exclusively to yolo"
```
