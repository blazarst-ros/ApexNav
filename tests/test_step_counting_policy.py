from pathlib import Path


def test_eval_configs_use_250_max_episode_steps():
    for config_path in [
        Path("config/habitat_eval_hm3dv1.yaml"),
        Path("config/habitat_eval_hm3dv2.yaml"),
        Path("config/habitat_eval_mp3d.yaml"),
    ]:
        assert "max_episode_steps: 250" in config_path.read_text(encoding="utf-8")


def test_eval_configs_keep_the_all_agents_perception_scheduler_budget():
    for config_path in [
        Path("config/habitat_eval_hm3dv1.yaml"),
        Path("config/habitat_eval_hm3dv2.yaml"),
        Path("config/habitat_eval_mp3d.yaml"),
    ]:
        text = config_path.read_text(encoding="utf-8")
        assert "perception_agents_per_step: 3" in text
        assert "perception_interval_steps: 1" in text


def test_habitat_evaluation_counts_every_executed_action():
    text = Path("habitat_evaluation.py").read_text(encoding="utf-8")
    assert "if agent_name in action_count_agents:" in text
    assert "single_action_counted = False" in text
    assert "if single_action_counted:" in text
    assert "if agent_name in movement_action_agents:\n                        ast[\"count_steps\"] += 1" not in text


def test_multiagent_runtime_log_prints_all_agent_step_counts():
    text = Path("habitat_evaluation.py").read_text(encoding="utf-8")

    assert "steps_str = \", \".join(" in text
    assert "f\"{name}={agent_states[name]['count_steps']}\"" in text
    assert "print(f\"\\n--------------Steps: {steps_str}--------------\")" in text
    assert "agent_states[agent_names[0]]['count_steps']" not in text


def test_readme_documents_step_counting_policy_and_max_steps():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "max_episode_steps: 250" in text
    assert "perception_agents_per_step: 3" in text
    assert "Every executed Habitat action increments the per-agent step counter." in text
