import re
from pathlib import Path


EVAL_CONFIGS = [
    Path("config/habitat_eval_hm3dv1.yaml"),
    Path("config/habitat_eval_hm3dv2.yaml"),
    Path("config/habitat_eval_mp3d.yaml"),
]


def _agent_block(text: str, agent_name: str) -> str:
    match = re.search(
        rf"      {agent_name}:\n(?P<body>.*?)(?=\n      agent_\d+:|\n\nnum_agents:)",
        text,
        re.S,
    )
    assert match, f"missing {agent_name}"
    return match.group("body")


def test_eval_configs_use_heterogeneous_agent_heights_and_success_distance():
    expected_heights = {
        "agent_0": 0.8,
        "agent_1": 1.5,
    }

    for config_path in EVAL_CONFIGS:
        text = config_path.read_text(encoding="utf-8")
        assert "success_distance: 0.2" in text
        for agent_name, expected_height in expected_heights.items():
            block = _agent_block(text, agent_name)
            assert f"height: {expected_height}" in block
            assert f"position: [0, {expected_height}, 0]" in block


def test_ros_publisher_uses_configured_camera_height_for_sensor_pose():
    text = Path("habitat2ros/habitat_publisher.py").read_text(encoding="utf-8")
    assert "camera_height: float = 0.88" in text
    assert "self.camera_height = camera_height" in text
    assert "gps[1] + self.camera_height" in text
    assert "gps[1] + 0.88" not in text
