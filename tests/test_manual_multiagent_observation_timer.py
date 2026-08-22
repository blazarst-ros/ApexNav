import unittest


class _FakeObservationPublisher:
    def __init__(self):
        self.observations = []

    def habitat_publish_ros_topic(self, observation):
        self.observations.append(observation)


class _FakeThresholdPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class ManualMultiAgentObservationTimerTests(unittest.TestCase):
    def test_timer_publishes_each_agents_latest_snapshot(self):
        import habitat_manual_control_multiagent as manual

        agent_0_pub = _FakeObservationPublisher()
        agent_1_pub = _FakeObservationPublisher()
        threshold_pub = _FakeThresholdPublisher()
        first_0 = {"rgb": "agent-0-frame-1", "depth": "agent-0-depth-1"}
        first_1 = {"rgb": "agent-1-frame-1", "depth": "agent-1-depth-1"}

        manual.ros_pubs = {"agent_0": agent_0_pub, "agent_1": agent_1_pub}
        manual.observation_snapshots = {
            "agent_0": first_0,
            "agent_1": first_1,
        }
        manual.confidence_threshold_pub = threshold_pub
        manual.fusion_threshold = 0.37

        manual.publish_observations(None)

        self.assertEqual(agent_0_pub.observations, [first_0])
        self.assertEqual(agent_1_pub.observations, [first_1])
        self.assertIsNot(agent_0_pub.observations[0], first_0)
        self.assertIsNot(agent_1_pub.observations[0], first_1)
        self.assertEqual([message.data for message in threshold_pub.messages], [0.37])

        latest_1 = {"rgb": "agent-1-frame-2", "depth": "agent-1-depth-2"}
        manual.observation_snapshots["agent_1"] = latest_1
        manual.publish_observations(None)

        self.assertEqual(agent_0_pub.observations[-1], first_0)
        self.assertEqual(agent_1_pub.observations[-1], latest_1)
        self.assertIsNot(agent_1_pub.observations[-1], latest_1)


if __name__ == "__main__":
    unittest.main()
