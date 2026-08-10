import ast
from pathlib import Path
import unittest


def load_selector():
    source = Path("vlm/utils/get_object_utils.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    selector_nodes = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_select_detector_labels"
    ]
    if not selector_nodes:
        return None

    namespace = {"COCO_CLASSES": {"chair", "couch"}}
    selector_module = ast.Module(body=selector_nodes, type_ignores=[])
    exec(compile(selector_module, "get_object_utils_selector", "exec"), namespace)
    return namespace["_select_detector_labels"]


class DetectorRoutingTests(unittest.TestCase):
    def test_coco_target_uses_yolo_when_similar_answer_is_not_coco(self):
        selector = load_selector()
        self.assertIsNotNone(selector)
        self.assertEqual(
            selector("chair", ["furniture"]),
            ("yolo", ["chair"]),
        )

    def test_non_coco_target_uses_grounding_dino_when_similar_answer_has_coco_labels(self):
        selector = load_selector()
        self.assertIsNotNone(selector)
        self.assertEqual(
            selector("ottoman", ["chair"]),
            ("gdino", ["ottoman", "chair"]),
        )

    def test_itm_helper_does_not_fallback_from_yolo_to_grounding_dino(self):
        source = Path("vlm/utils/get_object_utils.py").read_text(encoding="utf-8")
        start = source.index("def get_object_with_itm")
        helper = source[start:source.index("def crop_and_expand_box", start)]

        self.assertNotIn("using GroundingDINO for COCO label.", helper)
        self.assertIn("if not _is_yolov7_available():", helper)
