"""Ranker and inverse-optimization MVP pipeline for G-LIFT.

This module turns the five CP-SAT candidate schedules into candidate-level
features, compares them with the expert schedule, applies a transparent
rule-based Ranker, trains a small tree-based Ranker, and creates a conservative
inverse-optimization style weight table.

Important scope note:
The current MVP data is small.  The outputs should be described as an
"암묵지 학습 실증 개념" rather than a complete production-grade learned model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from .config import BASELINE_FEATURE_WEIGHTS, OUTPUTS_DIR, PROCESSED_DIR, REPORTS_DIR, SCENARIOS_DIR, SCENARIO_NAMES
from .feature_extractor import compute_candidate_feature_row
from .inverse_optimization import OBJECTIVE_FEATURES, learn_objective_weights
from .ranker_evaluation import ranking_report_for_group

CANDIDATE_FILE_NAMES = [
    "candidate_01_min_unserved.xlsx",
    "candidate_02_fairness.xlsx",
    "candidate_03_vehicle_preservation.xlsx",
    "candidate_04_fatigue.xlsx",
    "candidate_05_expert_style.xlsx",
]

# The rule-based Ranker weights are the shared expert-designed baseline.
RULE_WEIGHTS = BASELINE_FEATURE_WEIGHTS

MODEL_FEATURES = [
    "unserved_count",
    "weighted_unserved_priority",
    "hard_violation_count",
    "vehicle_fit_norm",
    "driver_fit_norm",
    "fairness_score",
    "fatigue_score",
    "vehicle_preservation_score",
    "other_handling_score",
    "other_assigned_count",
    "other_unserved_count",
    "diversity_score",
    "support_score",
    "weighted_support_score",
    "max_driver_minutes",
    "max_vehicle_minutes",
]


def _candidate_output_dir(scenario_name: str) -> Path:
    return OUTPUTS_DIR / "candidates" / scenario_name


def _read_excel_sheet(path: Path, sheet_name: str, fallback_empty: bool = True) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        if fallback_empty:
            return pd.DataFrame()
        raise


def _load_expert_schedule_for_scenario(scenario_name: str) -> pd.DataFrame:
    """Load the scenario expert schedule, falling back to processed baseline."""
    scenario_path = SCENARIOS_DIR / scenario_name / "expert_schedule.csv"
    processed_path = PROCESSED_DIR / "expert_schedule.csv"
    if scenario_path.exists():
        return pd.read_csv(scenario_path)
    if processed_path.exists():
        return pd.read_csv(processed_path)
    return pd.DataFrame()


def load_candidate_features(scenario_name: str) -> pd.DataFrame:
    """Build a feature matrix for all candidate schedules in one scenario."""
    out_dir = _candidate_output_dir(scenario_name)
    if not out_dir.exists():
        raise FileNotFoundError(f"Candidate output directory not found: {out_dir}. Run python -m src.candidate_generator --scenario {scenario_name} first.")
    expert_schedule = _load_expert_schedule_for_scenario(scenario_name)
    rows: List[Dict] = []
    for filename in CANDIDATE_FILE_NAMES:
        path = out_dir / filename
        if not path.exists():
            continue
        schedule = _read_excel_sheet(path, "schedule")
        summary = _read_excel_sheet(path, "score_summary")
        row = compute_candidate_feature_row(schedule, summary, expert_schedule, scenario_name=scenario_name)
        row["candidate_file"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def load_all_candidate_features(scenario_names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Build a feature matrix from every available scenario folder."""
    scenarios = list(scenario_names or SCENARIO_NAMES)
    frames = []
    for scenario in scenarios:
        out_dir = _candidate_output_dir(scenario)
        if not out_dir.exists():
            continue
        df = load_candidate_features(scenario)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def apply_rule_based_ranker(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add a transparent rule-based score used before ML training."""
    df = feature_df.copy()
    df["rule_based_score"] = 0.0
    for feature, weight in RULE_WEIGHTS.items():
        if feature not in df.columns:
            df[feature] = 0.0
        df["rule_based_score"] += weight * pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
    # Penalize any hard violation very strongly, even though candidates should have 0.
    if "hard_violation_count" in df.columns:
        df["rule_based_score"] -= 50.0 * pd.to_numeric(df["hard_violation_count"], errors="coerce").fillna(0.0)
    return df


def _build_estimator(model_type: str):
    """Build the configured tree-based regressor used for CV and final fit."""
    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=1)
    return GradientBoostingRegressor(random_state=42, n_estimators=120, max_depth=2, learning_rate=0.05)


def _cross_validated_metrics(estimator, X: pd.DataFrame, y: pd.Series) -> tuple[int, Optional[float], Optional[float]]:
    """Out-of-sample MAE/R2 via K-fold CV, robust to tiny datasets.

    Uses cross_val_predict so every row gets one held-out prediction, then
    scores the pooled out-of-fold predictions.  This is more stable than a
    single hold-out split on the small MVP dataset.  Returns
    (n_splits, mae_cv, r2_cv); if there are too few rows for a meaningful
    split, returns (0, None, None).
    """
    n = len(y)
    if n < 4:
        return 0, None, None
    n_splits = min(5, n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        oof_pred = cross_val_predict(estimator, X, y, cv=kf)
    except Exception:
        return n_splits, None, None
    mae_cv = float(mean_absolute_error(y, oof_pred))
    try:
        r2_cv = float(r2_score(y, oof_pred))
    except Exception:
        r2_cv = None
    return n_splits, mae_cv, r2_cv


def train_tree_ranker(feature_df: pd.DataFrame, model_type: str = "gradient_boosting") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train a small tree-based regressor and return scored data and reports.

    Target is expert_similarity_score.  The final model is fit on all available
    candidates (to score the candidates we actually have), while K-fold
    cross-validation provides out-of-sample MAE/R2 so generalization is not read
    off the in-sample fit.  The report explicitly labels this as proof-of-concept
    evidence given the small MVP dataset.
    """
    df = feature_df.copy()
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    X = df[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df.get("expert_similarity_score", pd.Series([0.0] * len(df))), errors="coerce").fillna(0.0)

    if len(df) < 3 or float(y.std()) < 1e-9:
        # Too little signal to train meaningfully.  Reuse the rule-based score
        # and create an explanatory report instead of failing.
        df["ml_ranker_score"] = df.get("rule_based_score", pd.Series([0.0] * len(df)))
        importance = pd.DataFrame({"feature": MODEL_FEATURES, "importance": [0.0] * len(MODEL_FEATURES)})
        report = pd.DataFrame([{
            "model_type": "not_trained_insufficient_signal",
            "training_rows": len(df),
            "cv_folds": 0,
            "mae_in_sample": None,
            "r2_in_sample": None,
            "mae_cv": None,
            "r2_cv": None,
            "note": "데이터가 너무 적거나 expert_similarity_score 변동이 작아, 규칙 기반 점수를 학습형 점수로 대체했습니다.",
        }])
        return df, importance, report

    model = _build_estimator(model_type)
    # Out-of-sample estimate before fitting the final model on all data.
    cv_folds, mae_cv, r2_cv = _cross_validated_metrics(model, X, y)
    model.fit(X, y)
    pred = model.predict(X)
    df["ml_ranker_score"] = pred

    if hasattr(model, "feature_importances_"):
        importance_values = list(model.feature_importances_)
    else:
        importance_values = [0.0] * len(MODEL_FEATURES)
    importance = pd.DataFrame({"feature": MODEL_FEATURES, "importance": importance_values}).sort_values("importance", ascending=False)
    try:
        r2 = float(r2_score(y, pred))
    except Exception:
        r2 = None
    if cv_folds:
        cv_note = f"{cv_folds}-fold 교차검증 out-of-sample 지표(mae_cv/r2_cv)를 함께 산출했습니다. 일반화 성능은 in-sample이 아닌 CV 기준으로 해석해야 합니다."
    else:
        cv_note = "행이 4개 미만이라 교차검증을 수행하지 못했습니다(mae_cv/r2_cv=None)."
    report = pd.DataFrame([{
        "model_type": model_type,
        "training_rows": len(df),
        "feature_count": len(MODEL_FEATURES),
        "cv_folds": cv_folds,
        "mae_in_sample": float(mean_absolute_error(y, pred)),
        "r2_in_sample": r2,
        "mae_cv": mae_cv,
        "r2_cv": r2_cv,
        "note": "MVP용 소규모 데이터 결과입니다. 생산 모델 성능으로 과장하지 않고 암묵지 학습 실증 개념으로 해석해야 합니다. " + cv_note,
    }])
    return df, importance, report


def build_before_after_comparison(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Create ranking comparison for optimizer/rule/ML/inverse scores."""
    df = scored_df.copy()
    # Optimizer order is candidate_01 -> candidate_05 unless score summary implies otherwise.
    candidate_order = {cid: i + 1 for i, cid in enumerate(["candidate_01", "candidate_02", "candidate_03", "candidate_04", "candidate_05"])}
    df["optimizer_rank"] = df["candidate_id"].map(candidate_order).fillna(999).astype(int)
    # Higher scores get smaller rank numbers.
    for score_col, rank_col in [
        ("rule_based_score", "rule_based_rank"),
        ("ml_ranker_score", "ml_ranker_rank"),
        ("learned_objective_score", "inverse_optimization_rank"),
        ("expert_similarity_score", "expert_similarity_rank"),
    ]:
        if score_col in df.columns:
            df[rank_col] = df.groupby("scenario_name")[score_col].rank(ascending=False, method="dense").astype(int)
    display_cols = [
        "scenario_name",
        "candidate_id",
        "candidate_profile",
        "optimizer_rank",
        "rule_based_rank",
        "ml_ranker_rank",
        "inverse_optimization_rank",
        "expert_similarity_rank",
        "rule_based_score",
        "ml_ranker_score",
        "learned_objective_score",
        "expert_similarity_score",
        "vehicle_match_rate",
        "driver_match_rate",
        "other_match_rate",
        "unserved_count",
        "hard_violation_count",
    ]
    return df[[c for c in display_cols if c in df.columns]].sort_values(["scenario_name", "ml_ranker_rank", "candidate_id"])


def save_ranker_outputs(
    feature_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    training_report: pd.DataFrame,
    learned_weights: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Save Ranker and inverse-optimization outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation").mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_matrix": output_dir / "candidate_feature_matrix.xlsx",
        "before_after": output_dir / "before_after_ranking_comparison.xlsx",
        "learned_weights": output_dir / "learned_objective_weights.xlsx",
        "training_report": output_dir / "ranker_training_report.xlsx",
        "feature_importance": output_dir / "ranker_feature_importance.xlsx",
    }
    feature_df.to_excel(paths["feature_matrix"], index=False)
    comparison.to_excel(paths["before_after"], index=False)
    learned_weights.to_excel(paths["learned_weights"], index=False)
    feature_importance.to_excel(paths["feature_importance"], index=False)

    eval_rows = []
    for scenario, group in scored_df.groupby("scenario_name"):
        rr = ranking_report_for_group(
            group,
            score_cols=["rule_based_score", "ml_ranker_score", "learned_objective_score"],
            target_col="expert_similarity_score",
        )
        if not rr.empty:
            rr.insert(0, "scenario_name", scenario)
            eval_rows.append(rr)
    ranking_eval = pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame()
    with pd.ExcelWriter(paths["training_report"], engine="xlsxwriter") as writer:
        training_report.to_excel(writer, sheet_name="model_report", index=False)
        ranking_eval.to_excel(writer, sheet_name="ranking_eval", index=False)
        scored_df.to_excel(writer, sheet_name="scored_features", index=False)
        feature_importance.to_excel(writer, sheet_name="feature_importance", index=False)
    # Requested report copy.
    report_copy = REPORTS_DIR / "evaluation" / "ranker_training_report.xlsx"
    with pd.ExcelWriter(report_copy, engine="xlsxwriter") as writer:
        training_report.to_excel(writer, sheet_name="model_report", index=False)
        ranking_eval.to_excel(writer, sheet_name="ranking_eval", index=False)
        comparison.to_excel(writer, sheet_name="before_after", index=False)
        learned_weights.to_excel(writer, sheet_name="learned_weights", index=False)
    paths["evaluation_training_report"] = report_copy
    return paths


def run_ranker_pipeline(scenario: str = "all", model_type: str = "gradient_boosting") -> Dict[str, object]:
    """Run feature extraction, rule Ranker, ML Ranker, and weight calibration."""
    if scenario == "all":
        feature_df = load_all_candidate_features()
        output_dir = OUTPUTS_DIR / "ranker" / "all_scenarios"
    else:
        feature_df = load_candidate_features(scenario)
        output_dir = OUTPUTS_DIR / "ranker" / scenario
    if feature_df.empty:
        raise FileNotFoundError("후보 배차표 feature를 만들 수 없습니다. 먼저 python -m src.candidate_generator --all-scenarios 를 실행하세요.")

    feature_df = apply_rule_based_ranker(feature_df)
    ml_df, importance, model_report = train_tree_ranker(feature_df, model_type=model_type)
    inv_result = learn_objective_weights(ml_df)
    scored = inv_result.scored_features.copy()
    # Preserve ML score and non-normalized fields from ml_df after inverse module normalization.
    for col in ["ml_ranker_score", "rule_based_score"]:
        if col in ml_df.columns:
            scored[col] = ml_df[col].values
    comparison = build_before_after_comparison(scored)
    paths = save_ranker_outputs(
        feature_df=feature_df,
        scored_df=scored,
        feature_importance=importance,
        training_report=model_report,
        learned_weights=inv_result.learned_weights,
        comparison=comparison,
        output_dir=output_dir,
    )
    return {
        "scenario": scenario,
        "feature_matrix": feature_df,
        "scored_features": scored,
        "feature_importance": importance,
        "training_report": model_report,
        "learned_weights": inv_result.learned_weights,
        "comparison": comparison,
        "output_paths": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G-LIFT Ranker and inverse-optimization MVP.")
    parser.add_argument("--scenario", default="all", help="Scenario name or 'all'.")
    parser.add_argument("--model", default="gradient_boosting", choices=["gradient_boosting", "random_forest"], help="Tree-based ranker model.")
    args = parser.parse_args()
    result = run_ranker_pipeline(scenario=args.scenario, model_type=args.model)
    print("G-LIFT Ranker/Inverse Optimization MVP completed.")
    print(f"scenario: {result['scenario']}")
    print("output files:")
    for name, path in result["output_paths"].items():
        print(f"- {name}: {path}")
    print("\nBefore/after ranking preview:")
    preview_cols = ["scenario_name", "candidate_id", "candidate_profile", "rule_based_rank", "ml_ranker_rank", "inverse_optimization_rank", "expert_similarity_score"]
    print(result["comparison"][[c for c in preview_cols if c in result["comparison"].columns]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
