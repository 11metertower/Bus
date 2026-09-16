"""Unit tests for learning-to-rank evaluation metrics."""

import math
import unittest

import pandas as pd

from src.ranker_evaluation import (
    dcg_at_k,
    ndcg_at_k,
    ranking_report_for_group,
    top1_match_rate,
)


class TestDcg(unittest.TestCase):
    def test_dcg_uses_log2_discount(self):
        # rel [3, 2]: 3/log2(2) + 2/log2(3) = 3 + 2/1.585...
        expected = 3.0 / math.log2(2) + 2.0 / math.log2(3)
        self.assertAlmostEqual(dcg_at_k([3, 2]), expected)

    def test_dcg_respects_k(self):
        self.assertAlmostEqual(dcg_at_k([3, 2, 1], k=1), 3.0)


class TestNdcg(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        self.assertAlmostEqual(ndcg_at_k([3, 2, 1]), 1.0)

    def test_all_zero_relevance_is_zero(self):
        self.assertEqual(ndcg_at_k([0, 0, 0]), 0.0)

    def test_reversed_ranking_less_than_one(self):
        self.assertLess(ndcg_at_k([1, 2, 3]), 1.0)


class TestTop1MatchRate(unittest.TestCase):
    def _df(self):
        return pd.DataFrame(
            {
                "candidate_id": ["candidate_01", "candidate_02"],
                "score": [0.9, 0.4],
                "expert_similarity_score": [80.0, 40.0],
            }
        )

    def test_match_when_top_ranked_is_top_target(self):
        self.assertEqual(top1_match_rate(self._df(), "score"), 1.0)

    def test_no_match_when_disagree(self):
        df = self._df()
        df["score"] = [0.4, 0.9]  # candidate_02 now ranks first
        self.assertEqual(top1_match_rate(df, "score"), 0.0)

    def test_empty_df_returns_zero(self):
        self.assertEqual(top1_match_rate(pd.DataFrame(), "score"), 0.0)


class TestRankingReportForGroup(unittest.TestCase):
    def test_report_has_one_row_per_score_column(self):
        df = pd.DataFrame(
            {
                "candidate_id": ["candidate_01", "candidate_02"],
                "candidate_profile": ["a", "b"],
                "rule_based_score": [0.9, 0.4],
                "ml_ranker_score": [0.5, 0.6],
                "expert_similarity_score": [80.0, 40.0],
            }
        )
        report = ranking_report_for_group(df, score_cols=["rule_based_score", "ml_ranker_score"])
        self.assertEqual(len(report), 2)
        self.assertEqual(set(report["score_column"]), {"rule_based_score", "ml_ranker_score"})
        self.assertIn("ndcg_at_5", report.columns)
        self.assertIn("top1_expert_match", report.columns)

    def test_empty_df_returns_empty_report(self):
        self.assertTrue(ranking_report_for_group(pd.DataFrame(), score_cols=["s"]).empty)


if __name__ == "__main__":
    unittest.main()
