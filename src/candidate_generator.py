"""Candidate schedule generation for G-LIFT MVP.

Baseline usage:
    python -m src.candidate_generator

Scenario usage:
    python -m src.scenario_generator
    python -m src.candidate_generator --scenario mixed_shortage
    python -m src.candidate_generator --all-scenarios
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import OUTPUTS_DIR, PROCESSED_DIR, REPORTS_DIR, SCENARIOS_DIR, SCENARIO_NAMES
from .optimizer import CandidateSolution, load_processed_inputs, solve_all_default_profiles

CANDIDATE_FILE_NAMES = {
    "candidate_01": "candidate_01_min_unserved.xlsx",
    "candidate_02": "candidate_02_fairness.xlsx",
    "candidate_03": "candidate_03_vehicle_preservation.xlsx",
    "candidate_04": "candidate_04_fatigue.xlsx",
    "candidate_05": "candidate_05_expert_style.xlsx",
}


def _clean_score_row(row: Dict) -> Dict:
    """Remove internal-only fields before export."""
    return {k: v for k, v in row.items() if not k.startswith("_")}


def resolve_input_dir(scenario_name: str = "baseline", input_dir: Optional[Path] = None) -> Path:
    """Return the processed CSV directory for a scenario."""
    if input_dir is not None:
        return Path(input_dir)
    if scenario_name == "baseline":
        scenario_dir = SCENARIOS_DIR / "baseline"
        return scenario_dir if scenario_dir.exists() else PROCESSED_DIR
    return SCENARIOS_DIR / scenario_name


def ensure_output_dirs(scenario_name: str) -> Path:
    scenario_output_dir = OUTPUTS_DIR / "candidates" / scenario_name
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation").mkdir(parents=True, exist_ok=True)
    return scenario_output_dir


def _assignment_key(row: pd.Series) -> tuple:
    if row.get("status") != "ASSIGNED":
        return (str(row.get("request_id")), "UNSERVED", "")
    if row.get("assignment_mode") == "OTHER_DIRECT_DRIVER":
        return (str(row.get("request_id")), str(row.get("assigned_vehicle_id")), "OTHER")
    return (str(row.get("request_id")), str(row.get("assigned_vehicle_id")), str(row.get("assigned_driver_id")))


def add_diversity_metrics(solutions: List[CandidateSolution]) -> None:
    """Add simple diversity summaries to each candidate score row."""
    previous_keys: set[tuple] = set()
    for idx, solution in enumerate(solutions):
        keys = {_assignment_key(row) for _, row in solution.schedule.iterrows()}
        if idx == 0 or not previous_keys:
            same = 0
            diversity_score = 100
        else:
            same = len(keys & previous_keys)
            denom = max(1, len(keys))
            diversity_score = int(round(100 * (1 - same / denom)))
        solution.score_breakdown["same_assignment_count_vs_previous"] = int(same)
        solution.score_breakdown["diversity_score_vs_previous"] = int(diversity_score)
        solution.score_breakdown["assignment_count_for_diversity"] = int(len(keys))
        previous_keys |= keys


def _scenario_metadata(input_dir: Path, scenario_name: str) -> Dict:
    path = Path(input_dir) / "scenario_metadata.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"scenario_name": scenario_name, "metadata_read_error": str(path)}
    return {"scenario_name": scenario_name, "description": "baseline 또는 metadata가 없는 입력입니다."}


def save_candidate_outputs(solutions: List[CandidateSolution], scenario_name: str = "baseline", input_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Save all candidate schedules and summary workbooks under a scenario folder."""
    scenario_output_dir = ensure_output_dirs(scenario_name)
    output_paths: Dict[str, Path] = {}
    score_rows = []
    reason_frames = []
    hard_frames = []

    add_diversity_metrics(solutions)

    for solution in solutions:
        candidate_id = solution.score_breakdown["candidate_id"]
        filename = CANDIDATE_FILE_NAMES.get(candidate_id, f"{candidate_id}.xlsx")
        path = scenario_output_dir / filename
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            solution.schedule.to_excel(writer, sheet_name="schedule", index=False)
            pd.DataFrame([_clean_score_row(solution.score_breakdown)]).to_excel(writer, sheet_name="score_summary", index=False)
            solution.unserved_reasons.to_excel(writer, sheet_name="unserved_reasons", index=False)
            solution.hard_constraint_check.to_excel(writer, sheet_name="hard_constraint_check", index=False)
        output_paths[candidate_id] = path
        row = _clean_score_row(solution.score_breakdown)
        row["scenario_name"] = scenario_name
        score_rows.append(row)
        if not solution.unserved_reasons.empty:
            reason_frames.append(solution.unserved_reasons)
        hard = solution.hard_constraint_check.copy()
        hard.insert(0, "candidate_id", candidate_id)
        hard.insert(0, "scenario_name", scenario_name)
        hard_frames.append(hard)

    score_breakdown = pd.DataFrame(score_rows)
    score_path = scenario_output_dir / "score_breakdown.xlsx"
    score_breakdown.to_excel(score_path, index=False)
    output_paths["score_breakdown"] = score_path

    if reason_frames:
        unserved_reasons = pd.concat(reason_frames, ignore_index=True)
    else:
        unserved_reasons = pd.DataFrame(columns=["candidate_id", "request_id", "unserved_reason_code", "unserved_reason_detail", "possible_relaxation"])
    reasons_path = scenario_output_dir / "unserved_reasons.xlsx"
    unserved_reasons.to_excel(reasons_path, index=False)
    output_paths["unserved_reasons"] = reasons_path

    hard_check = pd.concat(hard_frames, ignore_index=True) if hard_frames else pd.DataFrame()
    hard_path = scenario_output_dir / "hard_constraint_check.xlsx"
    hard_check.to_excel(hard_path, index=False)
    output_paths["hard_constraint_check"] = hard_path

    # Also save a copy under reports/evaluation for quick inspection.
    eval_path = REPORTS_DIR / "evaluation" / f"{scenario_name}_hard_constraint_check.xlsx"
    hard_check.to_excel(eval_path, index=False)
    output_paths["evaluation_hard_constraint_check"] = eval_path

    meta = _scenario_metadata(Path(input_dir) if input_dir is not None else resolve_input_dir(scenario_name), scenario_name)
    reason_counts = (
        unserved_reasons.groupby("unserved_reason_code").size().reset_index(name="count")
        if not unserved_reasons.empty and "unserved_reason_code" in unserved_reasons.columns
        else pd.DataFrame(columns=["unserved_reason_code", "count"])
    )
    summary_path = scenario_output_dir / "scenario_summary.xlsx"
    with pd.ExcelWriter(summary_path, engine="xlsxwriter") as writer:
        score_breakdown.to_excel(writer, sheet_name="score_breakdown", index=False)
        reason_counts.to_excel(writer, sheet_name="unserved_reason_counts", index=False)
        hard_check.to_excel(writer, sheet_name="hard_constraint_check", index=False)
        pd.DataFrame([meta]).to_excel(writer, sheet_name="metadata", index=False)
    output_paths["scenario_summary"] = summary_path
    return output_paths


def run_candidate_generation(
    processed_dir: Path = PROCESSED_DIR,
    scenario_name: str = "baseline",
    input_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Load inputs, solve five candidates, and save outputs."""
    input_dir = resolve_input_dir(scenario_name, input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}. Run python -m src.scenario_generator first, or use baseline.")
    tables = load_processed_inputs(input_dir)
    solutions = solve_all_default_profiles(tables)
    output_paths = save_candidate_outputs(solutions, scenario_name=scenario_name, input_dir=input_dir)
    score_breakdown = pd.DataFrame([_clean_score_row(s.score_breakdown) | {"scenario_name": scenario_name} for s in solutions])
    return {
        "scenario_name": scenario_name,
        "input_dir": input_dir,
        "solutions": solutions,
        "output_paths": output_paths,
        "score_breakdown": score_breakdown,
    }


def run_all_scenarios() -> Dict[str, Dict[str, object]]:
    """Run candidate generation for every known scenario folder that exists."""
    results: Dict[str, Dict[str, object]] = {}
    for scenario in SCENARIO_NAMES:
        input_dir = resolve_input_dir(scenario)
        if not input_dir.exists():
            continue
        results[scenario] = run_candidate_generation(scenario_name=scenario, input_dir=input_dir)
    # Combined summary workbook.
    if results:
        frames = [res["score_breakdown"] for res in results.values()]
        combined = pd.concat(frames, ignore_index=True)
        out = OUTPUTS_DIR / "candidates" / "all_scenarios_score_breakdown.xlsx"
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_excel(out, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate G-LIFT candidate schedules.")
    parser.add_argument("--scenario", default="baseline", help="Scenario name, e.g. baseline, vehicle_shortage, mixed_shortage")
    parser.add_argument("--input-dir", default=None, help="Optional custom processed CSV directory")
    parser.add_argument("--all-scenarios", action="store_true", help="Run every generated scenario folder")
    args = parser.parse_args()

    if args.all_scenarios:
        results = run_all_scenarios()
        print("G-LIFT candidate generation completed for all available scenarios.")
        for scenario, result in results.items():
            score_breakdown: pd.DataFrame = result["score_breakdown"]
            print(f"\n[{scenario}]")
            print(score_breakdown[["candidate_id", "candidate_profile", "solver_status", "assigned_count", "unserved_count", "hard_violation_count"]].to_string(index=False))
        return

    result = run_candidate_generation(scenario_name=args.scenario, input_dir=Path(args.input_dir) if args.input_dir else None)
    score_breakdown: pd.DataFrame = result["score_breakdown"]
    print(f"G-LIFT candidate generation completed. scenario={result['scenario_name']}")
    print(score_breakdown[["candidate_id", "candidate_profile", "solver_status", "assigned_count", "unserved_count", "hard_violation_count"]].to_string(index=False))
    print("\nOutput files:")
    for name, path in result["output_paths"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
