import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vlm.detector.yoloe import YOLOEClient
from vlm.detector.detections import ObjectDetections
from vlm.utils import get_itm_message as itm_module
from vlm.utils import get_object_utils as object_module
from vlm.utils.get_object_utils import get_object, get_object_class_names


class _FakeYOLOE:
    def predict(self, image, **kwargs):
        return SimpleNamespace(
            boxes=torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]),
            logits=torch.tensor([0.9, 0.7]),
            phrases=["chair", "table"],
            masks=[
                np.array([[1, 0], [0, 0]], dtype=np.uint8),
                np.array([[0, 0], [0, 1]], dtype=np.uint8),
            ],
        )


class _FakeITM:
    def infer(self, image, text):
        return {"response": 0.75, "itm score": 0.875, "timing": {"clip_ms": 4.0}}


class LiteVLMContractTests(unittest.TestCase):
    def test_yoloe_client_uses_lite_endpoint_and_decodes_masks(self):
        """Catches a client pointing at the legacy YOLOv7 endpoint or losing masks."""
        response = {
            "boxes": [[0.0, 0.0, 1.0, 1.0]],
            "logits": [0.9],
            "phrases": ["chair"],
            "masks": ["AQAAAQ=="],
        }
        client = YOLOEClient(port=12184)

        with patch("vlm.detector.yoloe.send_request", return_value=response) as request:
            detections = client.predict(
                np.zeros((2, 2, 3), dtype=np.uint8),
                classes=["chair"],
                conf_thres=0.3,
                iou_thres=0.5,
            )

        self.assertEqual(client.url, "http://localhost:12184/yoloe")
        self.assertEqual(detections.phrases, ["chair"])
        self.assertEqual(detections.masks[0].tolist(), [[1, 0], [0, 1]])
        self.assertEqual(request.call_args.kwargs["classes"], ["chair"])

    def test_class_names_keep_target_first_and_remove_duplicate_confusion_labels(self):
        """Catches target-index drift and duplicate class names in the ROS message."""
        self.assertEqual(
            get_object_class_names("chair | armchair", ["table", "chair", "table", " lamp ", ""]),
            ["chair", "table", "lamp"],
        )

    def test_get_object_preserves_default_shape_and_exposes_opt_in_timing(self):
        """Catches accidental fifth default return value or non-target label index zero."""
        cfg = SimpleNamespace(
            yoloe=SimpleNamespace(
                confidence_threshold=0.3,
                iou_threshold=0.5,
                agnostic_nms=True,
            )
        )
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with patch.object(object_module, "yoloe_detector", _FakeYOLOE()):
            default_result = get_object("chair | armchair", image, cfg, ["table"])
            stats_result = get_object(
                "chair | armchair", image, cfg, ["table"], return_stats=True
            )

        self.assertEqual(len(default_result), 4)
        self.assertAlmostEqual(default_result[1][0], 0.9)
        self.assertAlmostEqual(default_result[1][1], 0.7)
        self.assertEqual(default_result[3], [0, 1])
        self.assertEqual(default_result[2][0].shape, (2, 2))
        self.assertEqual(len(stats_result), 5)
        self.assertIn("yoloe_latency_ms", stats_result[4])

    def test_duplicate_target_aliases_do_not_reclassify_confusion_as_target(self):
        """Catches confusion labels shifted to index zero by repeated target aliases."""
        cfg = SimpleNamespace(
            yoloe=SimpleNamespace(
                confidence_threshold=0.3,
                iou_threshold=0.5,
                agnostic_nms=True,
            )
        )
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with patch.object(object_module, "yoloe_detector", _FakeYOLOE()):
            result = get_object("chair | chair", image, cfg, ["table"])

        self.assertEqual(get_object_class_names("chair | chair", ["table"]), ["chair", "table"])
        self.assertEqual(result[3], [0, 1])

    def test_confidence_filter_keeps_masks_aligned_with_detections(self):
        """Catches masks retaining positions from detections removed by confidence."""
        first_mask = np.array([[1]], dtype=np.uint8)
        second_mask = np.array([[2]], dtype=np.uint8)
        detections = ObjectDetections(
            boxes=torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]),
            logits=torch.tensor([0.2, 0.8]),
            phrases=["chair", "table"],
            masks=[first_mask, second_mask],
            image_source=None,
            fmt="xyxy",
        )

        detections.filter_by_conf(0.5)

        self.assertEqual(detections.phrases, ["table"])
        self.assertEqual([mask.tolist() for mask in detections.masks], [[[2]]])

    def test_class_filter_keeps_masks_aligned_with_detections(self):
        """Catches masks retaining positions from detections removed by class."""
        first_mask = np.array([[1]], dtype=np.uint8)
        second_mask = np.array([[2]], dtype=np.uint8)
        detections = ObjectDetections(
            boxes=torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]),
            logits=torch.tensor([0.2, 0.8]),
            phrases=["chair", "table"],
            masks=[first_mask, second_mask],
            image_source=None,
            fmt="xyxy",
        )

        detections.filter_by_class(["table"])

        self.assertEqual(detections.phrases, ["table"])
        self.assertEqual([mask.tolist() for mask in detections.masks], [[[2]]])

    def test_cosine_helper_returns_float_by_default_and_timing_only_on_request(self):
        """Catches a legacy BLIP tuple leaking into the cosine ROS topic contract."""
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with patch.object(itm_module, "itmclient", _FakeITM()):
            cosine = itm_module.get_itm_message_cosine(image, "chair", "everywhere")
            cosine_with_stats = itm_module.get_itm_message_cosine(
                image, "chair", "everywhere", return_stats=True
            )

        self.assertIsInstance(cosine, float)
        self.assertEqual(cosine, 0.75)
        self.assertEqual(cosine_with_stats, (0.75, {"clip_ms": 4.0}))


if __name__ == "__main__":
    unittest.main()
