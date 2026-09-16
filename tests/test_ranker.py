"""Unit tests for the rule-based and tree-based rankers, incl. cross-validation."""

import unittest

import numpy as np
import pandas as pd

from src.config import BASELINE_FEATURE_WEIGHTS
from src.ranker import (
    MODEL_FEATURES,
    RULE_WEIGHTS,
    _build_estimator,
    _cross_validated_metrics,
    apply_rule_based_ranker,
    train_tree_ranker,
)


def _feature_frame(n, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    data = {f: rng.normal(50, 10, n) for f in MODEL_FEATURES}
    df = pd.DataFrame(data)
    if signal:
        df["expert_similarity_score"] = 0.5 * df["support_score"] + rng.normal(0, 2, n)
    else:
        df["expert_similarity_score"] = 42.0  # zero variance -> insufficient signal
    df["candidate_id"] = [f"candidate_{i % 5 + 1:02d}" for i in range(n)]
    df["scenario_name"] = "s"
    return df


class TestSingleSourceOfTruth(unittest.TestCase):
    def test_rule_weights_is_the_shared_baseline(self):
        self.assertIs(RULE_WEIGHTS, BASELINE_FEATURE_WEIGHTS)


class TestApplyRuleBasedRanker(unittest.TestCase):
    def test_score_is_weighted_sum(self):
        # All features equal to 50 and weights sum to 1 -> score 50.
        df = pd.DataFrame([{k: 50.0 for k in RULE_WEIGHTS}])
        out = apply_rule_based_ranker(df)
        self.assertAlmostEqual(out["rule_based_score"].iloc[0], 50.0, places=6)

    def test_hard_violation_is_penalized(self):
        df = pd.DataFrame([{k: 50.0 for k in RULE_WEIGHTS}])
        df["hard_violation_count"] = [1]
        out = apply_rule_based_ranker(df)
        self.assertAlmostEqual(out["rule_based_score"].iloc[0], 0.0, places=6)  # 50 - 50*1


class TestCrossValidatedMetrics(unittest.TestCase):
    def test_too_few_rows_returns_none(self):
        df = _feature_frame(3)
        X = df[MODEL_FEATURES]
        y = df["expert_similarity_score"]
        folds, mae, r2 = _cross_validated_metrics(_build_estimator("gradient_boosting"), X, y)
        self.assertEqual(folds, 0)
        self.assertIsNone(mae)
        self.assertIsNone(r2)

    def test_enough_rows_produces_metrics(self):
        df = _feature_frame(12)
        X = df[MODEL_FEATURES]
        y = df["expert_similarity_score"]
        folds, mae, r2 = _cross_validated_metrics(_build_estimator("gradient_boosting"), X, y)
        self.assertEqual(folds, 5)  # min(5, 12)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(r2, float)


class TestTrainTreeRanker(unittest.TestCase):
    def test_insufficient_signal_falls_back_to_rule_score(self):
        df = apply_rule_based_ranker(_feature_frame(10, signal=False))
        scored, importance, report = train_tree_ranker(df)
        self.assertEqual(report["model_type"].iloc[0], "not_trained_insufficient_signal")
        self.assertIn("ml_ranker_score", scored.columns)
        # Report schema stays consistent with the trained branch.
        for col in ["cv_folds", "mae_cv", "r2_cv"]:
            self.assertIn(col, report.columns)

    def test_trained_report_has_in_sample_and_cv_metrics(self):
        df = apply_rule_based_ranker(_feature_frame(20, signal=True))
        scored, importance, report = train_tree_ranker(df)
        row = report.iloc[0]
        self.assertEqual(row["model_type"], "gradient_boosting")
        self.assertGreater(row["cv_folds"], 0)
        self.assertIsNotNone(row["mae_in_sample"])
        self.assertIsNotNone(row["mae_cv"])
        self.assertIn("r2_cv", report.columns)
        self.assertIn("ml_ranker_score", scored.columns)
        self.assertEqual(len(importance), len(MODEL_FEATURES))

    def test_random_forest_model_type(self):
        df = apply_rule_based_ranker(_feature_frame(20, signal=True))
        _, _, report = train_tree_ranker(df, model_type="random_forest")
        self.assertEqual(report["model_type"].iloc[0], "random_forest")


if __name__ == "__main__":
    unittest.main()
