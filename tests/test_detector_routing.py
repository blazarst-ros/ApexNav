import ast
from pathlib import Path
import unittest


def load_label_merger():
    source = Path("vlm/utils/get_object_utils.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    selector_nodes = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_merge_labels"
    ]
    if not selector_nodes:
        return None

    namespace = {}
    selector_module = ast.Module(body=selector_nodes, type_ignores=[])
    exec(compile(selector_module, "get_object_utils_selector", "exec"), namespace)
    return namespace["_merge_labels"]


class DetectorRoutingTests(unittest.TestCase):
    def test_coco_target_and_confusion_labels_share_yoloe_request(self):
        selector = load_label_merger()
        self.assertIsNotNone(selector)
        self.assertEqual(
            selector("chair", ["furniture"]),
            (["chair"], ["chair", "furniture"]),
        )

    def test_non_coco_target_and_coco_confusion_labels_share_yoloe_request(self):
        selector = load_label_merger()
        self.assertIsNotNone(selector)
        self.assertEqual(
            selector("ottoman", ["chair"]),
            (["ottoman"], ["ottoman", "chair"]),
        )

    def test_object_helper_has_no_legacy_detector_fallback(self):
        source = Path("vlm/utils/get_object_utils.py").read_text(encoding="utf-8")

        self.assertIn("from vlm.detector.yoloe import YOLOEClient", source)
        self.assertNotIn("GroundingDINO", source)
        self.assertNotIn("YOLOv7", source)
