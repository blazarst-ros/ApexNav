from pathlib import Path


FSM_HEADER = Path("src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h")
FSM_SOURCE = Path("src/planner/exploration_manager/src/exploration_fsm.cpp")
DATA_HEADER = Path("src/planner/exploration_manager/include/exploration_manager/exploration_data.h")
MANAGER_HEADER = Path("src/planner/exploration_manager/include/exploration_manager/exploration_manager.h")
MANAGER_SOURCE = Path("src/planner/exploration_manager/src/exploration_manager.cpp")


def test_exploration_strategy_topics_publish_structured_state():
    header = FSM_HEADER.read_text(encoding="utf-8")
    source = FSM_SOURCE.read_text(encoding="utf-8")
    data_header = DATA_HEADER.read_text(encoding="utf-8")

    assert "exploration_strategy_pub_" in header
    assert '"/ros/agent_0/exploration_strategy"' in source
    assert '"/ros/agent_1/exploration_strategy"' not in source
    assert "NUM_STRATEGY_TOPIC_AGENTS" not in header
    assert "void publishExplorationStrategy();" in header
    assert "void ExplorationFSM::publishExplorationStrategy()" in source
    assert 'std_msgs::String msg;' in source
    assert '\\"mode\\"' in source
    assert '\\"target_type\\"' in source
    assert '\\"target_id\\"' in source
    assert '\\"semantic_score\\"' in source
    assert '\\"path_length\\"' in source
    assert '\\"target_pos\\"' in source
    assert "struct NavigationStrategyInfo" in data_header
    assert 'mode = "UNKNOWN";' in data_header
    assert 'target_type = "NONE";' in data_header
    assert "strategy_infos_" in data_header


def test_exploration_strategy_records_non_frontier_modes():
    manager_header = MANAGER_HEADER.read_text(encoding="utf-8")
    manager_source = MANAGER_SOURCE.read_text(encoding="utf-8")

    assert "void setStrategyInfo(" in manager_header
    assert "void ExplorationManager::setStrategyInfo(" in manager_source
    assert 'setStrategyInfo("PLANNING", "NONE"' in manager_source
    assert 'setStrategyInfo("SEARCH_BEST_OBJECT", "OBJECT"' in manager_source
    assert 'setStrategyInfo("SEARCH_OVER_DEPTH_OBJECT", "OBJECT"' in manager_source
    assert 'setStrategyInfo("SEARCH_SUSPICIOUS_OBJECT", "OBJECT"' in manager_source
    assert 'setStrategyInfo("NO_COVERABLE_FRONTIER", "NONE"' in manager_source
    assert 'setStrategyInfo("NO_PASSABLE_FRONTIER", "NONE"' in manager_source
    assert 'setStrategyInfo("GEOMETRIC_FRONTIER", "FRONTIER"' in manager_source
