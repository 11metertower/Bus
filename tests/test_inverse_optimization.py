"""Unit tests for the inverse-optimization weight calibration."""

import unittest

import pandas as pd

from src.config import BASELINE_FEATURE_WEIGHTS
from src.inverse_optimization import (
    INITIAL_WEIGHTS,
    OBJECTIVE_FEATURES,
    _normalize_series,
    learn_objective_weights,
    prepare_weight_features,
)


class TestNormalizeSeries(unittest.TestCase):
    def test_min_max_scaling(self):
        out = _normalize_series(pd.Series([0.0, 5.0, 10.0]))
        self.assertEqual(out.tolist(), [0.0, 0.5, 1.0])

    def test_constant_series_maps_to_zero(self):
        out = _normalize_series(pd.Series([7.0, 7.0, 7.0]))
        self.assertEqual(out.tolist(), [0.0, 0.0, 0.0])


class TestPrepareWeightFeatures(unittest.TestCase):
    def test_missing_features_are_added_and_normalized(self):
        df = prepare_weight_features(pd.DataFrame({"support_score": [10.0, 20.0]}))
        for col in OBJECTIVE_FEATURES:
            self.assertIn(col, df.columns)
        self.assertIn("expert_similarity_target", df.columns)


class TestLearnObjectiveWeights(unittest.TestCase):
    def _feature_df(self):
        rows = []
        for i in range(5):
            row = {col: float(i + 1) for col in OBJECTIVE_FEATURES}
            row["expert_similarity_score"] = 20.0 * (i + 1)
            rows.append(row)
        return pd.DataFrame(rows)

    def test_learned_weights_sum_to_one(self):
        result = learn_objective_weights(self._feature_df())
        self.assertAlmostEqual(result.learned_weights["learned_weight"].sum(), 1.0, places=6)

    def test_weights_are_nonnegative(self):
        result = learn_objective_weights(self._feature_df())
        self.assertTrue((result.learned_weights["learned_weight"] >= 0).all())

    def test_report_columns_present(self):
        result = learn_objective_weights(self._feature_df())
        for col in ["feature", "initial_weight", "data_signal_weight", "learned_weight", "interpretation"]:
            self.assertIn(col, result.learned_weights.columns)

    def test_scored_features_have_objective_scores(self):
        result = learn_objective_weights(self._feature_df())
        self.assertIn("learned_objective_score", result.scored_features.columns)
        self.assertIn("initial_objective_score", result.scored_features.columns)


class TestSingleSourceOfTruth(unittest.TestCase):
    def test_initial_weights_is_the_shared_baseline(self):
        self.assertIs(INITIAL_WEIGHTS, BASELINE_FEATURE_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
