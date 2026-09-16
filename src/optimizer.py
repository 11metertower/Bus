"""OR-Tools CP-SAT optimizer for the G-LIFT MVP.

The optimizer consumes standardized CSV/DataFrames produced by
``python -m src.preprocess``.  It deliberately keeps the first MVP model
transparent rather than clever: sparse variables are generated only for
candidate assignments that already satisfy class, availability, maintenance,
and driver-skill filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from ortools.sat.python import cp_model

from .config import PROCESSED_DIR, TIME_SLOT_MINUTES
from .explain import assignment_explanation, diagnose_unserved_reason
from .hard_constraint_checker import hard_violation_total, run_hard_constraint_checks
from .feature_extractor import (
    as_key,
    build_availability_index,
    driver_fit_score,
    driver_has_required_skill,
    is_resource_available,
    make_expert_maps,
    request_slots,
    summarize_candidate,
    vehicle_fit_score,
    vehicle_matches_request,
)


@dataclass(frozen=True)
class OptimizerProfile:
    """Objective weights for generating one candidate schedule."""

    candidate_id: str
    profile_name: str
    description: str
    unserved_base_penalty: int = 100_000
    unserved_priority_penalty: int = 20_000
    vehicle_fit_weight: int = 10
    driver_fit_weight: int = 10
    max_driver_minutes_weight: int = 0
    max_vehicle_minutes_weight: int = 0
    expert_vehicle_bonus: int = 0
    expert_driver_bonus: int = 0
    solver_time_limit_seconds: float = 20.0
    diversity_penalty_weight: int = 0


@dataclass
class CandidateSolution:
    """Container returned by the optimizer for one candidate profile."""

    schedule: pd.DataFrame
    score_breakdown: Dict[str, Any]
    unserved_reasons: pd.DataFrame
    hard_constraint_check: pd.DataFrame


DEFAULT_PROFILES: List[OptimizerProfile] = [
    OptimizerProfile(
        candidate_id="candidate_01",
        profile_name="미지원 최소화형",
        description="가능한 많은 배차 요청을 지원하는 안",
        unserved_base_penalty=160_000,
        unserved_priority_penalty=35_000,
        vehicle_fit_weight=8,
        driver_fit_weight=8,
        max_driver_minutes_weight=1,
        max_vehicle_minutes_weight=1,
    ),
    OptimizerProfile(
        candidate_id="candidate_02",
        profile_name="운전병 공정성 중심형",
        description="운전병별 당일 운행시간 최대값을 낮추는 안",
        unserved_base_penalty=130_000,
        unserved_priority_penalty=28_000,
        vehicle_fit_weight=6,
        driver_fit_weight=8,
        max_driver_minutes_weight=80,
        max_vehicle_minutes_weight=1,
    ),
    OptimizerProfile(
        candidate_id="candidate_03",
        profile_name="차량 보존 중심형",
        description="특정 차량 과사용을 줄이고 차량 운행시간 최대값을 낮추는 안",
        unserved_base_penalty=130_000,
        unserved_priority_penalty=28_000,
        vehicle_fit_weight=5,
        driver_fit_weight=5,
        max_driver_minutes_weight=3,
        max_vehicle_minutes_weight=70,
    ),
    OptimizerProfile(
        candidate_id="candidate_04",
        profile_name="피로도 최소화형",
        description="운전병 장시간 운행 부담을 줄이는 안",
        unserved_base_penalty=130_000,
        unserved_priority_penalty=28_000,
        vehicle_fit_weight=5,
        driver_fit_weight=8,
        max_driver_minutes_weight=100,
        max_vehicle_minutes_weight=2,
    ),
    OptimizerProfile(
        candidate_id="candidate_05",
        profile_name="배차반장 스타일 모방형",
        description="배차반장 베스트표와 차량/운전병 선택 패턴이 유사한 안",
        unserved_base_penalty=130_000,
        unserved_priority_penalty=28_000,
        vehicle_fit_weight=7,
        driver_fit_weight=7,
        max_driver_minutes_weight=5,
        max_vehicle_minutes_weight=5,
        expert_vehicle_bonus=2_000,
        expert_driver_bonus=2_500,
    ),
]


def load_processed_inputs(processed_dir: Path = PROCESSED_DIR) -> Dict[str, pd.DataFrame]:
    """Load standardized CSV inputs created by the preprocessing module."""
    processed_dir = Path(processed_dir)
    names = {
        "requests": "requests.csv",
        "vehicles": "vehicles.csv",
        "drivers": "drivers.csv",
        "vehicle_availability": "vehicle_availability.csv",
        "driver_availability": "driver_availability.csv",
        "expert_schedule": "expert_schedule.csv",
    }
    tables: Dict[str, pd.DataFrame] = {}
    missing = []
    for key, filename in names.items():
        path = processed_dir / filename
        if not path.exists():
            missing.append(str(path))
        else:
            tables[key] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError("Missing processed input files:\n" + "\n".join(missing))
    return tables


def _clean_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Normalize join keys after CSV loading."""
    out = {k: v.copy() for k, v in tables.items()}
    out["requests"]["request_id"] = out["requests"]["request_id"].astype(str)
    out["requests"]["required_vehicle_class"] = out["requests"]["required_vehicle_class"].astype(str)
    out["vehicles"]["vehicle_id"] = out["vehicles"]["vehicle_id"].map(as_key)
    out["vehicles"]["vehicle_number"] = out["vehicles"].get("vehicle_number", out["vehicles"]["vehicle_id"]).map(as_key)
    out["drivers"]["driver_id"] = out["drivers"]["driver_id"].astype(str)
    out["vehicle_availability"]["resource_id"] = out["vehicle_availability"]["resource_id"].map(as_key)
    out["driver_availability"]["driver_id"] = out["driver_availability"]["driver_id"].astype(str)
    out["expert_schedule"]["request_id"] = out["expert_schedule"]["request_id"].astype(str)
    out["expert_schedule"]["assigned_vehicle_id"] = out["expert_schedule"]["assigned_vehicle_id"].map(as_key)
    return out


def _build_candidate_assignment_lists(
    requests: pd.DataFrame,
    vehicles: pd.DataFrame,
    drivers: pd.DataFrame,
    vehicle_availability_index: Dict[str, set],
    driver_availability_index: Dict[str, set],
    expert_vehicle_map: Dict[str, str],
    expert_driver_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate sparse feasible x/y assignment candidates.

    Returns:
        driver_candidates: each item corresponds to x_request_vehicle_driver.
        other_candidates: each item corresponds to y_request_vehicle_for_other.
    """
    vehicle_rows = [v for _, v in vehicles.iterrows()]
    driver_rows = [d for _, d in drivers.iterrows()]
    driver_candidates: List[Dict[str, Any]] = []
    other_candidates: List[Dict[str, Any]] = []

    for _, request in requests.iterrows():
        request_id = str(request.get("request_id"))
        slots = request_slots(request.get("start_min"), request.get("end_min"))
        is_other = str(request.get("driver_mode")) == "OTHER_DIRECT_DRIVER" or request.get("requires_driver") is False
        expert_vehicle_id = expert_vehicle_map.get(request_id)
        expert_driver_name = expert_driver_map.get(request_id)
        for vehicle in vehicle_rows:
            vehicle_id = as_key(vehicle.get("vehicle_id"))
            if vehicle_id is None:
                continue
            if bool(vehicle.get("is_maintenance_or_unavailable", False)):
                continue
            if not vehicle_matches_request(vehicle, request):
                continue
            if not is_resource_available(vehicle_id, slots, vehicle_availability_index):
                continue
            v_score = vehicle_fit_score(vehicle, request, expert_vehicle_id)
            v_match = bool(expert_vehicle_id and vehicle_id == expert_vehicle_id)
            base = {
                "request_id": request_id,
                "vehicle_id": vehicle_id,
                "vehicle_number": as_key(vehicle.get("vehicle_number")) or vehicle_id,
                "slots": slots,
                "duration_min": int(request.get("duration_min") or 0),
                "vehicle_fit_score": v_score,
                "expert_vehicle_match": v_match,
                "request": request,
                "vehicle": vehicle,
            }
            if is_other:
                other_candidates.append(base)
                continue
            for driver in driver_rows:
                driver_id = str(driver.get("driver_id"))
                if not driver_has_required_skill(driver, vehicle, request):
                    continue
                if not is_resource_available(driver_id, slots, driver_availability_index):
                    continue
                d_score = driver_fit_score(driver, vehicle, request, expert_driver_name)
                d_match = bool(expert_driver_name and str(driver.get("driver_name")) == str(expert_driver_name))
                cand = dict(base)
                cand.update({
                    "driver_id": driver_id,
                    "driver_name": str(driver.get("driver_name")),
                    "driver_fit_score": d_score,
                    "expert_driver_match": d_match,
                    "driver": driver,
                })
                driver_candidates.append(cand)
    return driver_candidates, other_candidates


def solve_candidate(
    tables: Dict[str, pd.DataFrame],
    profile: OptimizerProfile,
    previous_assignment_keys: Optional[set] = None,
) -> CandidateSolution:
    """Solve one candidate profile with OR-Tools CP-SAT."""
    tables = _clean_tables(tables)
    requests = tables["requests"]
    vehicles = tables["vehicles"]
    drivers = tables["drivers"]
    expert_schedule = tables["expert_schedule"]

    vehicle_availability_index = build_availability_index(tables["vehicle_availability"], "resource_id")
    driver_availability_index = build_availability_index(tables["driver_availability"], "driver_id")
    expert_vehicle_map, expert_driver_map = make_expert_maps(expert_schedule)

    driver_candidates, other_candidates = _build_candidate_assignment_lists(
        requests, vehicles, drivers,
        vehicle_availability_index, driver_availability_index,
        expert_vehicle_map, expert_driver_map,
    )

    model = cp_model.CpModel()
    x_vars: Dict[int, cp_model.IntVar] = {}
    y_vars: Dict[int, cp_model.IntVar] = {}
    unserved: Dict[str, cp_model.IntVar] = {}

    for i, cand in enumerate(driver_candidates):
        x_vars[i] = model.NewBoolVar(f"x_{cand['request_id']}_{cand['vehicle_id']}_{cand['driver_id']}")
    for i, cand in enumerate(other_candidates):
        y_vars[i] = model.NewBoolVar(f"y_{cand['request_id']}_{cand['vehicle_id']}")
    for _, req in requests.iterrows():
        rid = str(req.get("request_id"))
        unserved[rid] = model.NewBoolVar(f"unserved_{rid}")

    # Assignment: exactly one supported assignment or unserved for every request.
    x_by_request: Dict[str, List[cp_model.IntVar]] = {}
    y_by_request: Dict[str, List[cp_model.IntVar]] = {}
    for i, cand in enumerate(driver_candidates):
        x_by_request.setdefault(cand["request_id"], []).append(x_vars[i])
    for i, cand in enumerate(other_candidates):
        y_by_request.setdefault(cand["request_id"], []).append(y_vars[i])

    for _, req in requests.iterrows():
        rid = str(req.get("request_id"))
        if str(req.get("driver_mode")) == "OTHER_DIRECT_DRIVER" or req.get("requires_driver") is False:
            model.Add(sum(y_by_request.get(rid, [])) + unserved[rid] == 1)
        else:
            model.Add(sum(x_by_request.get(rid, [])) + unserved[rid] == 1)

    # Vehicle time conflicts.
    vehicle_slot_vars: Dict[Tuple[str, int], List[cp_model.IntVar]] = {}
    for i, cand in enumerate(driver_candidates):
        for slot in cand["slots"]:
            vehicle_slot_vars.setdefault((cand["vehicle_id"], int(slot)), []).append(x_vars[i])
    for i, cand in enumerate(other_candidates):
        for slot in cand["slots"]:
            vehicle_slot_vars.setdefault((cand["vehicle_id"], int(slot)), []).append(y_vars[i])
    for vars_for_slot in vehicle_slot_vars.values():
        if len(vars_for_slot) > 1:
            model.Add(sum(vars_for_slot) <= 1)

    # Driver time conflicts.  Other variables are intentionally excluded.
    driver_slot_vars: Dict[Tuple[str, int], List[cp_model.IntVar]] = {}
    for i, cand in enumerate(driver_candidates):
        for slot in cand["slots"]:
            driver_slot_vars.setdefault((cand["driver_id"], int(slot)), []).append(x_vars[i])
    for vars_for_slot in driver_slot_vars.values():
        if len(vars_for_slot) > 1:
            model.Add(sum(vars_for_slot) <= 1)

    # Workload variables for first-pass fairness/fatigue and vehicle preservation.
    total_duration = int(requests["duration_min"].fillna(0).sum())
    driver_minutes: Dict[str, cp_model.IntVar] = {}
    active_driver_ids = sorted({cand["driver_id"] for cand in driver_candidates})
    for driver_id in active_driver_ids:
        terms = [int(cand["duration_min"]) * x_vars[i] for i, cand in enumerate(driver_candidates) if cand["driver_id"] == driver_id]
        var = model.NewIntVar(0, total_duration, f"driver_minutes_{driver_id}")
        model.Add(var == (sum(terms) if terms else 0))
        driver_minutes[driver_id] = var
    max_driver_minutes = model.NewIntVar(0, total_duration, "max_driver_minutes")
    if driver_minutes:
        model.AddMaxEquality(max_driver_minutes, list(driver_minutes.values()))
    else:
        model.Add(max_driver_minutes == 0)

    vehicle_minutes: Dict[str, cp_model.IntVar] = {}
    active_vehicle_ids = sorted({cand["vehicle_id"] for cand in driver_candidates} | {cand["vehicle_id"] for cand in other_candidates})
    for vehicle_id in active_vehicle_ids:
        terms = [int(cand["duration_min"]) * x_vars[i] for i, cand in enumerate(driver_candidates) if cand["vehicle_id"] == vehicle_id]
        terms += [int(cand["duration_min"]) * y_vars[i] for i, cand in enumerate(other_candidates) if cand["vehicle_id"] == vehicle_id]
        var = model.NewIntVar(0, total_duration, f"vehicle_minutes_{vehicle_id}")
        model.Add(var == (sum(terms) if terms else 0))
        vehicle_minutes[vehicle_id] = var
    max_vehicle_minutes = model.NewIntVar(0, total_duration, "max_vehicle_minutes")
    if vehicle_minutes:
        model.AddMaxEquality(max_vehicle_minutes, list(vehicle_minutes.values()))
    else:
        model.Add(max_vehicle_minutes == 0)

    objective_terms = []

    # Unserved request penalty.
    for _, req in requests.iterrows():
        rid = str(req.get("request_id"))
        priority = int(req.get("priority_initial") or 1)
        objective_terms.append((profile.unserved_base_penalty + profile.unserved_priority_penalty * priority) * unserved[rid])

    # Soft assignment scores and expert-style bonuses.  Scores are subtracted in a minimization model.
    for i, cand in enumerate(driver_candidates):
        score = profile.vehicle_fit_weight * int(cand.get("vehicle_fit_score", 0))
        score += profile.driver_fit_weight * int(cand.get("driver_fit_score", 0))
        if cand.get("expert_vehicle_match"):
            score += profile.expert_vehicle_bonus
        if cand.get("expert_driver_match"):
            score += profile.expert_driver_bonus
        key = (cand["request_id"], cand["vehicle_id"], cand.get("driver_id"))
        if previous_assignment_keys and key in previous_assignment_keys:
            score -= profile.diversity_penalty_weight
        objective_terms.append(-score * x_vars[i])

    for i, cand in enumerate(other_candidates):
        score = profile.vehicle_fit_weight * int(cand.get("vehicle_fit_score", 0))
        if cand.get("expert_vehicle_match"):
            score += profile.expert_vehicle_bonus
        key = (cand["request_id"], cand["vehicle_id"], "OTHER")
        if previous_assignment_keys and key in previous_assignment_keys:
            score -= profile.diversity_penalty_weight
        objective_terms.append(-score * y_vars[i])

    objective_terms.append(profile.max_driver_minutes_weight * max_driver_minutes)
    objective_terms.append(profile.max_vehicle_minutes_weight * max_vehicle_minutes)
    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = profile.solver_time_limit_seconds
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    schedule_rows: List[Dict[str, Any]] = []
    chosen_keys: set = set()
    selected_x_by_request: Dict[str, Dict[str, Any]] = {}
    selected_y_by_request: Dict[str, Dict[str, Any]] = {}
    for i, cand in enumerate(driver_candidates):
        if solver.Value(x_vars[i]) == 1:
            selected_x_by_request[cand["request_id"]] = cand
            chosen_keys.add((cand["request_id"], cand["vehicle_id"], cand.get("driver_id")))
    for i, cand in enumerate(other_candidates):
        if solver.Value(y_vars[i]) == 1:
            selected_y_by_request[cand["request_id"]] = cand
            chosen_keys.add((cand["request_id"], cand["vehicle_id"], "OTHER"))

    unserved_reason_rows: List[Dict[str, Any]] = []
    for _, req in requests.iterrows():
        rid = str(req.get("request_id"))
        base = {
            "candidate_id": profile.candidate_id,
            "candidate_profile": profile.profile_name,
            "request_id": rid,
            "request_type": req.get("request_type"),
            "required_vehicle_class": req.get("required_vehicle_class"),
            "destination": req.get("destination"),
            "requesting_unit": req.get("requesting_unit"),
            "purpose": req.get("purpose"),
            "start_min": req.get("start_min"),
            "end_min": req.get("end_min"),
            "start_time": req.get("start_time"),
            "end_time": req.get("end_time"),
            "duration_min": req.get("duration_min"),
            "priority_initial": req.get("priority_initial"),
            "is_other": bool(req.get("driver_mode") == "OTHER_DIRECT_DRIVER" or req.get("requires_driver") is False),
            "requires_driver": not bool(req.get("driver_mode") == "OTHER_DIRECT_DRIVER" or req.get("requires_driver") is False),
        }
        if rid in selected_x_by_request:
            cand = selected_x_by_request[rid]
            row = dict(base)
            row.update({
                "status": "ASSIGNED",
                "assignment_mode": "DRIVER_REQUIRED",
                "assigned_vehicle_id": cand["vehicle_id"],
                "assigned_vehicle_number": cand["vehicle_number"],
                "assigned_driver_id": cand["driver_id"],
                "assigned_driver_name": cand["driver_name"],
                "vehicle_fit_score": cand["vehicle_fit_score"],
                "driver_fit_score": cand["driver_fit_score"],
                "expert_vehicle_match": cand.get("expert_vehicle_match", False),
                "expert_driver_match": cand.get("expert_driver_match", False),
                "unserved_reason_code": "",
                "unserved_reason_detail": "",
                "possible_relaxation": "",
            })
        elif rid in selected_y_by_request:
            cand = selected_y_by_request[rid]
            row = dict(base)
            row.update({
                "status": "ASSIGNED",
                "assignment_mode": "OTHER_DIRECT_DRIVER",
                "assigned_vehicle_id": cand["vehicle_id"],
                "assigned_vehicle_number": cand["vehicle_number"],
                "assigned_driver_id": "",
                "assigned_driver_name": "Other 직접운전",
                "vehicle_fit_score": cand["vehicle_fit_score"],
                "driver_fit_score": 0,
                "expert_vehicle_match": cand.get("expert_vehicle_match", False),
                "expert_driver_match": False,
                "unserved_reason_code": "",
                "unserved_reason_detail": "",
                "possible_relaxation": "",
            })
        else:
            reason = diagnose_unserved_reason(req, vehicles, drivers, vehicle_availability_index, driver_availability_index)
            row = dict(base)
            row.update({
                "status": "UNSERVED",
                "assignment_mode": "UNSERVED",
                "assigned_vehicle_id": "",
                "assigned_vehicle_number": "",
                "assigned_driver_id": "",
                "assigned_driver_name": "",
                "vehicle_fit_score": 0,
                "driver_fit_score": 0,
                "expert_vehicle_match": False,
                "expert_driver_match": False,
                "unserved_reason_code": reason["reason_code"],
                "unserved_reason_detail": reason["reason_detail"],
                "possible_relaxation": reason["possible_relaxation"],
            })
            reason_row = dict(row)
            unserved_reason_rows.append(reason_row)
        row["explanation"] = assignment_explanation(pd.Series(row))
        schedule_rows.append(row)

    schedule = pd.DataFrame(schedule_rows)
    unserved_reasons = pd.DataFrame(unserved_reason_rows)
    hard_check = run_hard_constraint_checks(schedule, tables)
    summary = summarize_candidate(schedule)
    score_breakdown = {
        "candidate_id": profile.candidate_id,
        "candidate_profile": profile.profile_name,
        "description": profile.description,
        "solver_status": status_name,
        "objective_value": float(solver.ObjectiveValue()) if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
        "max_driver_minutes": int(solver.Value(max_driver_minutes)) if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
        "max_vehicle_minutes": int(solver.Value(max_vehicle_minutes)) if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
        "x_variable_count": len(x_vars),
        "y_variable_count": len(y_vars),
        "unserved_variable_count": len(unserved),
        "hard_violation_count": hard_violation_total(hard_check),
        **summary,
        "_chosen_keys": chosen_keys,  # internal use; removed before exporting
    }
    return CandidateSolution(schedule, score_breakdown, unserved_reasons, hard_check)


def solve_all_default_profiles(tables: Dict[str, pd.DataFrame]) -> List[CandidateSolution]:
    """Solve the five MVP candidate profiles."""
    solutions: List[CandidateSolution] = []
    previous_keys: set = set()
    for profile in DEFAULT_PROFILES:
        solution = solve_candidate(tables, profile, previous_assignment_keys=previous_keys)
        solutions.append(solution)
        previous_keys |= set(solution.score_breakdown.get("_chosen_keys", set()))
    return solutions
