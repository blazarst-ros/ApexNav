from pathlib import Path

from basic_utils import path_utils


def test_scene_runtime_output_dir_uses_three_character_scene_prefix():
    output_dir_builder = getattr(path_utils, "scene_runtime_output_dir", None)

    assert callable(output_dir_builder)
    assert output_dir_builder(
        "videos/test_hm3dv2_multiagent_val",
        "/data/scene_datasets/hm3d/val/TbHJrupSAjP/TbHJrupSAjP.glb",
    ) == Path("videos/test_hm3dv2_multiagent_val/TbH")
    assert output_dir_builder(
        "videos/test_hm3dv2_multiagent_val",
        "/data/scene_datasets/hm3d/val/2azQ1b91cZZ/2azQ1b91cZZ.glb",
    ) == Path("videos/test_hm3dv2_multiagent_val/2az")
