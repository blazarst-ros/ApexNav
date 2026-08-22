import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from vlm.detector import yoloe


class _CachingModel:
    def __init__(self):
        self.get_text_pe_calls = []
        self.set_classes_calls = []
        self.predictor = None

    def get_text_pe(self, classes):
        self.get_text_pe_calls.append(tuple(classes))
        return ("text-pe", tuple(classes))

    def set_classes(self, classes, text_pe):
        self.set_classes_calls.append((tuple(classes), text_pe))


class _LegacySetClassesModel(_CachingModel):
    def set_classes(self, classes, *args):
        if args:
            raise TypeError("legacy set_classes accepts labels only")
        self.set_classes_calls.append((tuple(classes), None))


class _PersistentSettings(dict):
    def __init__(self):
        super().__init__()
        self.persistent_writes = 0

    def __setitem__(self, key, value):
        self.persistent_writes += 1
        super().__setitem__(key, value)


def _detector_with_model(model):
    detector = yoloe.YOLOEDetector.__new__(yoloe.YOLOEDetector)
    detector.model = model
    detector._cached_classes = ()
    detector._cached_text_pe = None
    detector._supports_text_pe_cache = hasattr(model, "get_text_pe")
    return detector


class YOLOEClassCacheTests(unittest.TestCase):
    def test_normalized_labels_reuse_text_pe_until_the_tuple_changes(self):
        model = _CachingModel()
        detector = _detector_with_model(model)

        first = detector._normalize_classes([" chair ", "table", "chair", ""])
        detector._apply_classes(first)
        detector._apply_classes(detector._normalize_classes(["chair", "table"]))
        detector._apply_classes(detector._normalize_classes(["chair", "sofa"]))

        self.assertEqual(first, ("chair", "table"))
        self.assertEqual(
            model.get_text_pe_calls,
            [("chair", "table"), ("chair", "sofa")],
        )
        self.assertEqual(len(model.set_classes_calls), 3)

    def test_legacy_set_classes_fallback_disables_text_pe_cache(self):
        model = _LegacySetClassesModel()
        detector = _detector_with_model(model)

        detector._apply_classes(("chair",))
        detector._apply_classes(("sofa",))

        self.assertEqual(model.get_text_pe_calls, [("chair",)])
        self.assertFalse(detector._supports_text_pe_cache)
        self.assertEqual(
            model.set_classes_calls,
            [(("chair",), None), (("sofa",), None)],
        )

    def test_uninitialized_predictor_is_reset_before_setting_classes(self):
        model = _LegacySetClassesModel()
        model.predictor = SimpleNamespace(model=None)
        detector = _detector_with_model(model)
        detector._supports_text_pe_cache = False

        detector._apply_classes(("chair",))

        self.assertIsNone(model.predictor)
        self.assertEqual(model.set_classes_calls, [(("chair",), None)])

    def test_asset_cache_update_does_not_trigger_settings_persistence(self):
        settings = _PersistentSettings()
        fake_utils = ModuleType("ultralytics.utils")
        fake_utils.SETTINGS = settings

        with tempfile.TemporaryDirectory() as temp_dir:
            model_cache = Path(temp_dir) / "model-cache"
            (model_cache / "ultralytics").mkdir(parents=True)
            with patch.object(
                yoloe,
                "_conda_model_cache",
                return_value=model_cache / "yoloe-11l-seg.pt",
            ), patch.dict(sys.modules, {"ultralytics.utils": fake_utils}):
                yoloe._configure_ultralytics_asset_cache()

        self.assertEqual(settings["weights_dir"], str(model_cache / "ultralytics"))
        self.assertEqual(settings.persistent_writes, 0)


if __name__ == "__main__":
    unittest.main()
