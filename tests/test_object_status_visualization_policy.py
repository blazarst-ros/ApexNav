from pathlib import Path


MAP_ROS_CPP = Path("src/planner/plan_env/src/map_ros.cpp")


def _visualization_source() -> str:
    text = MAP_ROS_CPP.read_text(encoding="utf-8")
    return text[text.index("void MapROS::publishObjectVisualizations()") :]


def _marker_text_source() -> str:
    text = _visualization_source()
    return text[text.index("std::ostringstream marker_text") : text.index("marker.text =")]


def _table_text_source() -> str:
    text = _visualization_source()
    return text[text.index("size_t display_row_count") : text.index("std_msgs::Header image_header")]


def test_map_marker_shows_cluster_id_and_compact_state_suffix():
    text = _marker_text_source()
    compact = " ".join(text.split())

    assert 'marker_text << "C" << std::setfill(\'0\') << std::setw(3) << snapshot.cluster_id' in compact
    assert "state_suffix" in compact
    assert "marker.scale.z = 0.16" in _visualization_source()


def test_rviz_table_shows_raw_confidence_and_score_for_each_label():
    text = _table_text_source()
    compact = " ".join(text.split())

    assert "display_row_count" in text
    assert "for (const auto& snapshot : snapshots)" in text
    assert "for (const auto& label : snapshot.labels)" in text
    assert '<< "C" << std::setfill(\'0\') << std::setw(3) << snapshot.cluster_id' in compact
    assert '"Cluster       Label       Obs              Confidence       Score"' in text
    assert "<< label.label_index" in compact
    assert "label.detection_count" in text
    assert "label.confidence" in text
    assert "label.evidence_points * label.confidence" in text
    assert "state_name" not in text
    assert "best_name" not in text
    assert "label_name" not in text


def test_episode_reset_clears_cluster_markers_and_status_table():
    text = MAP_ROS_CPP.read_text(encoding="utf-8")

    reset = text[text.index("void MapROS::resetEpisodeState()") : text.index(
        "void MapROS::detectedObjectCloudCallback"
    )]
    visualization = _visualization_source()
    assert "cluster_markers_need_reset_ = true" in reset
    assert "publishObjectVisualizations()" in reset
    assert "visualization_msgs::Marker::DELETEALL" in visualization
