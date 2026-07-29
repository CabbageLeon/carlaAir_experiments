import math
import os
import unittest
from unittest.mock import patch

import numpy as np

from experiments.spf_eval.metrics import (
    EscortEpisode,
    LandingEpisode,
    escort_summary,
    landing_summary,
    recovery_time,
)
from experiments.spf_eval.prompts import escort_prompt, landing_prompt
from experiments.spf_eval.spf_policy import DASHSCOPE_COMPATIBLE_BASE_URL, SPFPolicy


class SPFMetricTests(unittest.TestCase):
    def test_landing_summary_is_seed_based(self):
        c0 = [
            LandingEpisode(1, 3.0, True, True),
            LandingEpisode(2, 3.0, False, False),
        ]
        c1 = [
            LandingEpisode(1, 3.0, False, False),
            LandingEpisode(2, 3.0, True, True),
        ]
        summary = landing_summary(c1, c0)
        self.assertEqual(summary["TSR"], 1.0)
        self.assertEqual(summary["LSR"], 0.5)
        self.assertEqual(summary["CCR"], 0.5)
        self.assertEqual(summary["CG"], 0.0)

    def test_recovery_requires_half_second_sustained_iou(self):
        samples = [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2)]
        self.assertAlmostEqual(recovery_time(samples, 0.0), 0.2)
        self.assertEqual(recovery_time([(0.2, 0.2), (0.6, 0.0)], 0.0), 15.0)

    def test_escort_summary_uses_capped_rat(self):
        summary = escort_summary(
            [
                EscortEpisode(1, 1, 2, (1.0, 15.0)),
                EscortEpisode(2, 1, 1, (5.0,)),
            ]
        )
        self.assertAlmostEqual(summary["RSR"], 0.75)
        self.assertAlmostEqual(summary["RAT"], 6.5)


class SPFProtocolTests(unittest.TestCase):
    def test_prompts_match_modes(self):
        self.assertNotIn("Assistant hint", landing_prompt("C0", "left", "moving", "approach"))
        self.assertIn("cargo bed is forward-left", landing_prompt("C2", "forward-left", "moving slowly", "approach"))
        self.assertIn("temporarily occluded", escort_prompt("C1", True, "forward-right", "continues forward", "occlusion recovery"))

    def test_spf_response_parser(self):
        self.assertEqual(SPFPolicy.parse_point('```json\n[{"point": [750, 250], "depth": 7}]\n```'), (250.0, 750.0, 7.0))
        with self.assertRaises(ValueError):
            SPFPolicy.parse_point('[{"point": [5, 5], "depth": 11}]')

    def test_spf_prompt_names_the_visual_target(self):
        prompt = SPFPolicy._prompt("Follow the moving truck and keep it in view.")
        self.assertIn("Target object: the moving truck", prompt)
        self.assertIn('"point": [x, y]', prompt)
        self.assertIn("do not return an empty list", prompt)

    def test_spf_uses_the_fixed_compatible_endpoint(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://invalid.example/v1"}):
            policy = SPFPolicy.from_environment()
        self.assertEqual(policy.config.base_url, DASHSCOPE_COMPATIBLE_BASE_URL)


if __name__ == "__main__":
    unittest.main()
