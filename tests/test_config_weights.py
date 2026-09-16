"""Unit tests for the canonical baseline feature-weight table."""

import unittest

from src.config import BASELINE_FEATURE_WEIGHTS
from src.inverse_optimization import INITIAL_WEIGHTS, OBJECTIVE_FEATURES
from src.ranker import RULE_WEIGHTS


class TestBaselineFeatureWeights(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(BASELINE_FEATURE_WEIGHTS.values()), 1.0, places=9)

    def test_has_all_objective_features(self):
        self.assertEqual(set(BASELINE_FEATURE_WEIGHTS), set(OBJECTIVE_FEATURES))

    def test_weights_are_nonnegative(self):
        self.assertTrue(all(w >= 0 for w in BASELINE_FEATURE_WEIGHTS.values()))

    def test_single_source_of_truth(self):
        # Both consumers must reference the exact same object, so they can never
        # drift apart again.
        self.assertIs(RULE_WEIGHTS, BASELINE_FEATURE_WEIGHTS)
        self.assertIs(INITIAL_WEIGHTS, BASELINE_FEATURE_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
