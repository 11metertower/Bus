"""Simple inverse-optimization style weight calibration for G-LIFT.

This module does not claim to recover a definitive human objective function.
With the current small MVP dataset, it estimates a transparent set of objective
weights that make candidates with higher expert-similarity receive higher
scores.  The purpose is to demonstrate the mechanism: expert schedule ->
feature differences -> calibrated objective weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


OBJECTIVE_FEATURES = [
    "support_score",
    "weighted_support_score",
    "vehicle_fit_norm",
    "driver_fit_norm",
    "fairness_score",
    "fatigue_score",
    "vehicle_preservation_score",
    "other_handling_score",
    "expert_vehicle_match_rate",
    "expert_driver_match_rate",
    "expert_other_match_rate",
    "diversity_score",
]

INITIAL_WEIGHTS = {
    "support_score": 0.22,
    "weighted_support_score": 0.16,
    "vehicle_fit_norm": 0.09,
    "driver_fit_norm": 0.09,
    "fairness_score": 0.09,
    "fatigue_score": 0.08,
    "vehicle_preservation_score": 0.07,
    "other_handling_score": 0.06,
    "expert_vehicle_match_rate": 0.05,
    "expert_driver_match_rate": 0.05,
    "expert_other_match_rate": 0.03,
    "diversity_score": 0.01,
}


@dataclass
class InverseOptimizationResult:
    """Container for calibrated weights and scored feature matrix."""

    learned_weights: pd.DataFrame
    scored_features: pd.DataFrame


def _normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series to 0~1."""
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return pd.Series([0.0] * len(x), index=s.index)
    return (x - lo) / (hi - lo)


def prepare_weight_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized feature table for inverse-weight calibration."""
    df = feature_df.copy()
    for col in OBJECTIVE_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = _normalize_series(df[col])
    if "expert_similarity_score" not in df.columns:
        df["expert_similarity_score"] = 0.0
    df["expert_similarity_target"] = _normalize_series(df["expert_similarity_score"])
    return df


def learn_objective_weights(feature_df: pd.DataFrame) -> InverseOptimizationResult:
    """Estimate nonnegative objective weights from expert-similarity targets.

    Algorithm:
    1. Normalize objective features to 0~1.
    2. Measure each feature's positive association with expert similarity.
    3. Blend that data-driven signal with a stable initial expert weight.
    4. Normalize weights to sum to 1.

    This is deliberately conservative for small data.  It avoids overclaiming
    and creates an interpretable weight table for the competition demo.
    """
    df = prepare_weight_features(feature_df)
    target = pd.to_numeric(df["expert_similarity_target"], errors="coerce").fillna(0.0)

    raw_scores: Dict[str, float] = {}
    for col in OBJECTIVE_FEATURES:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if len(df) < 2 or float(x.std()) < 1e-9 or float(target.std()) < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, target)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        # Keep only positive association; negative features should not receive
        # a positive learned weight in this simple MVP formulation.
        raw_scores[col] = max(0.0, corr)

    raw_total = sum(raw_scores.values())
    if raw_total <= 1e-9:
        data_weights = {k: 1.0 / len(OBJECTIVE_FEATURES) for k in OBJECTIVE_FEATURES}
    else:
        data_weights = {k: raw_scores[k] / raw_total for k in OBJECTIVE_FEATURES}

    # Conservative blend: keep the original expert-designed objective visible.
    blend_data = 0.55
    blended = {}
    for col in OBJECTIVE_FEATURES:
        blended[col] = blend_data * data_weights[col] + (1 - blend_data) * INITIAL_WEIGHTS.get(col, 0.0)
    total = sum(blended.values()) or 1.0
    learned = {k: v / total for k, v in blended.items()}

    df["learned_objective_score"] = 0.0
    df["initial_objective_score"] = 0.0
    for col in OBJECTIVE_FEATURES:
        df["learned_objective_score"] += learned[col] * df[col]
        df["initial_objective_score"] += INITIAL_WEIGHTS.get(col, 0.0) * df[col]

    weights = pd.DataFrame([
        {
            "feature": col,
            "initial_weight": INITIAL_WEIGHTS.get(col, 0.0),
            "data_signal_weight": data_weights[col],
            "learned_weight": learned[col],
            "interpretation": _interpret_weight(col),
        }
        for col in OBJECTIVE_FEATURES
    ]).sort_values("learned_weight", ascending=False)
    return InverseOptimizationResult(learned_weights=weights, scored_features=df)


def _interpret_weight(feature: str) -> str:
    explanations = {
        "support_score": "미지원 없이 최대한 많은 배차를 살리는 기준",
        "weighted_support_score": "중요도가 높은 배차를 우선 살리는 기준",
        "vehicle_fit_norm": "배차 특성에 맞는 차량을 고르는 기준",
        "driver_fit_norm": "기량·숙련도를 고려한 운전병 선택 기준",
        "fairness_score": "운전병 운행 부담을 균등하게 배분하는 기준",
        "fatigue_score": "당일 운행시간 과부하를 줄이는 기준",
        "vehicle_preservation_score": "특정 차량 과사용을 줄이는 기준",
        "other_handling_score": "Other 직접운전 배차를 차량-only로 정확히 처리하는 기준",
        "expert_vehicle_match_rate": "배차반장표 차량 선택과의 유사성",
        "expert_driver_match_rate": "배차반장표 운전병 선택과의 유사성",
        "expert_other_match_rate": "배차반장표 Other 처리와의 유사성",
        "diversity_score": "후보안 간 차별성을 확보하는 기준",
    }
    return explanations.get(feature, "후보 배차표 품질 기준")
