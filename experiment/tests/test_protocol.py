from __future__ import annotations

import unittest

from poclib.protocol import EXPECTED_EPISODES_PER_MODEL, expected_keys, iter_episode_specs


class ProtocolTests(unittest.TestCase):
    def test_cardinality_and_uniqueness(self) -> None:
        specs = list(iter_episode_specs())
        self.assertEqual(len(specs), EXPECTED_EPISODES_PER_MODEL)
        self.assertEqual(len(expected_keys()), EXPECTED_EPISODES_PER_MODEL)

    def test_exactly_twenty_preregistered_videos(self) -> None:
        videos = [spec for spec in iter_episode_specs() if spec.record_video]
        self.assertEqual(len(videos), 20)
        self.assertEqual({(spec.init_state_id, spec.sampling_seed) for spec in videos}, {(0, 1234), (25, 1235)})


if __name__ == "__main__":
    unittest.main()
