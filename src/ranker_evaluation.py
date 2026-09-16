"""Evaluation helpers for G-LIFT Ranker MVP.

The functions in this module intentionally use small, transparent metrics.
The current project data is a proof-of-concept dataset, so the metrics are
used to demonstrate the evaluation structure rather than to claim a fully
trained production model.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import math
import pandas as pd


def dcg_at_k(relevances: Sequence[float], k: int = 5) -> float:
    """Discounted cumulative gain for a ranked list.

    Higher relevance near the top gets more credit.  This is a standard
    learning-to-rank metric, but the implementation is kept short for MVP use.
    """
    score = 0.0
    for i, rel in enumerate(list(relevances)[:k], start=1):
        score += float(rel) / math.log2(i + 1)
    return score


def ndcg_at_k(relevances: Sequence[float], k: int = 5) -> float:
    """Normalized DCG@k.  Returns 0 when every relevance is 0."""
    actual = dcg_at_k(relevances, k=k)
    ideal = dcg_at_k(sorted([float(x) for x in relevances], reverse=True), k=k)
    if ideal <= 0:
        return 0.0
    return actual / ideal


def top1_match_rate(df: pd.DataFrame, rank_col: str, target_col: str = "expert_similarity_score") -> float:
    """Return 1.0 if the top-ranked row is also the highest-target row."""
    if df.empty or rank_col not in df.columns or target_col not in df.columns:
        return 0.0
    top_by_rank = df.sort_values(rank_col, ascending=False).iloc[0].get("candidate_id")
    top_by_target = df.sort_values(target_col, ascending=False).iloc[0].get("candidate_id")
    return 1.0 if top_by_rank == top_by_target else 0.0


def ranking_report_for_group(df: pd.DataFrame, score_cols: Iterable[str], target_col: str = "expert_similarity_score") -> pd.DataFrame:
    """Create compact per-score ranking evaluation rows for one scenario."""
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    for col in score_cols:
        if col not in df.columns:
            continue
        ranked = df.sort_values(col, ascending=False)
        rows.append({
            "score_column": col,
            "ndcg_at_5": ndcg_at_k(ranked[target_col].fillna(0).tolist(), k=5) if target_col in ranked.columns else 0.0,
            "top1_expert_match": top1_match_rate(df, col, target_col=target_col),
            "top_candidate_id": ranked.iloc[0].get("candidate_id"),
            "top_candidate_profile": ranked.iloc[0].get("candidate_profile"),
        })
    return pd.DataFrame(rows)
