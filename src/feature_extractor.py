"""Feature and scoring helpers for the G-LIFT Optimizer MVP.

This module deliberately keeps the first-version scoring rules simple.  The
CP-SAT optimizer must first guarantee hard constraints.  The scores here are
soft preferences used to choose among feasible assignments.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .config import TIME_SLOT_MINUTES
from .utils import normalize_vehicle_number


def to_bool(value: Any) -> bool:
    """Convert common CSV boolean representations to a bool."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "t", "yes", "y"}


def as_key(value: Any) -> Optional[str]:
    """Normalize IDs that may be read as 2202, 2202.0, or '2202'."""
    if value is None or pd.isna(value):
        return None
    text = normalize_vehicle_number(value)
    return None if text in {None, ""} else str(text)


def request_slots(start_min: Any, end_min: Any, slot_minutes: int = TIME_SLOT_MINUTES) -> List[int]:
    """Return occupied discrete slot indices for [start_min, end_min).

    The end is exclusive: a request ending at 10:00 and another starting at
    10:00 do not overlap.
    """
    if pd.isna(start_min) or pd.isna(end_min):
        return []
    start = int(start_min)
    end = int(end_min)
    if end <= start:
        return []
    return list(range(start // slot_minutes, (end + slot_minutes - 1) // slot_minutes))


def infer_required_scope_level(request_type: Any) -> int:
    """Map request type to required driving scope.

    Lower numbers mean higher capability in the uploaded skill code system:
    1=long-distance outside base, 2=short-distance outside base, 3=inside base,
    4=not available.
    """
    text = "" if pd.isna(request_type) else str(request_type)
    if "장거리" in text:
        return 1
    if "단거리" in text:
        return 2
    return 3


def infer_skill_slot(required_skill_text: Any, vehicle_class: Any = None) -> str:
    """Choose which driver skill column should be checked."""
    text = "" if pd.isna(required_skill_text) else str(required_skill_text)
    vclass = "" if vehicle_class is None or pd.isna(vehicle_class) else str(vehicle_class)
    if "대형" in text:
        return "skill_large"
    if "중형" in text:
        return "skill_medium"
    if "소형" in text:
        return "skill_small"
    # Fallback from vehicle class labels.
    if any(word in vclass for word in ["버스", "9톤", "5톤", "구난", "트렉터"]):
        return "skill_large"
    if any(word in vclass for word in ["1톤", "승합", "지프"]):
        return "skill_medium"
    return "skill_small"


def driver_has_required_skill(driver: pd.Series, vehicle: pd.Series, request: pd.Series) -> bool:
    """Return True if the driver can operate the vehicle for the request scope."""
    if not to_bool(driver.get("skill_valid", True)):
        return False
    slot = infer_skill_slot(vehicle.get("required_skill_text"), vehicle.get("vehicle_class"))
    level = driver.get(slot)
    try:
        level = int(level)
    except Exception:
        return False
    required_level = infer_required_scope_level(request.get("request_type"))
    return 1 <= level <= required_level


def vehicle_matches_request(vehicle: pd.Series, request: pd.Series) -> bool:
    """Hard vehicle compatibility for MVP: exact standardized vehicle class."""
    rv = request.get("required_vehicle_class")
    vv = vehicle.get("vehicle_class")
    if pd.isna(rv) or pd.isna(vv):
        return False
    return str(rv).strip() == str(vv).strip()


def build_availability_index(avail: pd.DataFrame, id_col: str) -> Dict[str, Set[int]]:
    """Return {resource_id: set(slot_index where available)}."""
    if avail.empty:
        return {}
    df = avail.copy()
    df[id_col] = df[id_col].map(as_key)
    df = df[df["is_available"].map(to_bool)]
    result: Dict[str, Set[int]] = {}
    for rid, group in df.groupby(id_col):
        if rid is None or pd.isna(rid):
            continue
        result[str(rid)] = set(group["slot_index"].astype(int).tolist())
    return result


def is_resource_available(resource_id: Any, slots: Sequence[int], availability_index: Dict[str, Set[int]]) -> bool:
    """Check that every occupied slot is available for a resource."""
    key = as_key(resource_id)
    if key is None:
        return False
    available = availability_index.get(str(key), set())
    return bool(slots) and all(slot in available for slot in slots)


def vehicle_fit_score(vehicle: pd.Series, request: pd.Series, expert_vehicle_id: Optional[str] = None) -> int:
    """Simple 0-100 vehicle soft score among already-compatible vehicles."""
    score = 70
    requested_vehicle = as_key(request.get("requested_vehicle_id"))
    vehicle_id = as_key(vehicle.get("vehicle_id"))
    request_type = "" if pd.isna(request.get("request_type")) else str(request.get("request_type"))
    vehicle_class = "" if pd.isna(vehicle.get("vehicle_class")) else str(vehicle.get("vehicle_class"))

    if requested_vehicle and vehicle_id == requested_vehicle:
        score += 15
    # Expert-match bonuses are handled explicitly by the expert-style profile.
    if "장거리" in request_type and any(word in vehicle_class for word in ["고속버스", "대형", "5톤", "9톤"]):
        score += 8
    if "정기" in request_type and "버스" in vehicle_class:
        score += 5
    if bool(vehicle.get("is_rental", False)):
        score -= 3
    if bool(vehicle.get("is_electric", False)) and "장거리" in request_type:
        score -= 3
    return max(0, min(100, int(score)))


def driver_fit_score(driver: pd.Series, vehicle: pd.Series, request: pd.Series, expert_driver_name: Optional[str] = None) -> int:
    """Simple 0-100 driver soft score among already-feasible drivers."""
    score = 70
    slot = infer_skill_slot(vehicle.get("required_skill_text"), vehicle.get("vehicle_class"))
    required_level = infer_required_scope_level(request.get("request_type"))
    try:
        level = int(driver.get(slot))
    except Exception:
        level = 4

    # Exact capability match is good.  Over-qualified drivers are useful but
    # should be preserved for harder work, so give a slightly smaller bonus.
    if level == required_level:
        score += 15
    elif level < required_level:
        score += 8
    else:
        score -= 50

    # Expert-match bonuses are handled explicitly by the expert-style profile.
    return max(0, min(100, int(score)))


def make_expert_maps(expert_schedule: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build request_id -> expert vehicle/driver maps."""
    if expert_schedule is None or expert_schedule.empty:
        return {}, {}
    vehicle_map: Dict[str, str] = {}
    driver_map: Dict[str, str] = {}
    for _, row in expert_schedule.iterrows():
        rid = str(row.get("request_id"))
        vid = as_key(row.get("assigned_vehicle_id"))
        drv = row.get("assigned_driver_raw")
        if vid:
            vehicle_map[rid] = vid
        if drv is not None and not pd.isna(drv):
            driver_map[rid] = str(drv)
    return vehicle_map, driver_map


def summarize_candidate(schedule: pd.DataFrame) -> Dict[str, Any]:
    """Compute lightweight summary metrics from a candidate schedule."""
    if schedule.empty:
        return {
            "assigned_count": 0,
            "unserved_count": 0,
            "other_assigned_count": 0,
            "vehicle_fit_score_sum": 0,
            "driver_fit_score_sum": 0,
            "expert_vehicle_match_count": 0,
            "expert_driver_match_count": 0,
        }
    assigned = schedule[schedule["status"] == "ASSIGNED"]
    return {
        "assigned_count": int((schedule["status"] == "ASSIGNED").sum()),
        "unserved_count": int((schedule["status"] == "UNSERVED").sum()),
        "other_assigned_count": int(((assigned["assignment_mode"] == "OTHER_DIRECT_DRIVER")).sum()),
        "vehicle_fit_score_sum": int(assigned.get("vehicle_fit_score", pd.Series(dtype=int)).fillna(0).sum()),
        "driver_fit_score_sum": int(assigned.get("driver_fit_score", pd.Series(dtype=int)).fillna(0).sum()),
        "expert_vehicle_match_count": int(assigned.get("expert_vehicle_match", pd.Series(dtype=bool)).fillna(False).sum()),
        "expert_driver_match_count": int(assigned.get("expert_driver_match", pd.Series(dtype=bool)).fillna(False).sum()),
    }

# ---------------------------------------------------------------------------
# Ranker-level feature extraction helpers
# ---------------------------------------------------------------------------

def _safe_num(value: Any, default: float = 0.0) -> float:
    """Return numeric value while tolerating blanks from Excel exports."""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_rate(numerator: float, denominator: float) -> float:
    """Return a percentage-like 0~100 rate."""
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def compute_expert_similarity_metrics(schedule: pd.DataFrame, expert_schedule: pd.DataFrame) -> Dict[str, float]:
    """Compare one candidate schedule with the expert schedule.

    The expert schedule is treated as an expert demonstration, not as an
    absolute truth.  These metrics are therefore used as Ranker labels and
    explanatory signals, not as hard pass/fail judgments.
    """
    if schedule is None or schedule.empty or expert_schedule is None or expert_schedule.empty:
        return {
            "vehicle_match_count": 0,
            "vehicle_match_rate": 0.0,
            "driver_match_count": 0,
            "driver_match_rate": 0.0,
            "other_match_count": 0,
            "other_match_rate": 0.0,
            "unserved_match_count": 0,
            "unserved_match_rate": 0.0,
            "expert_similarity_score": 0.0,
        }

    sched = schedule.copy()
    exp = expert_schedule.copy()
    sched["request_id"] = sched["request_id"].astype(str)
    exp["request_id"] = exp["request_id"].astype(str)
    exp_vehicle_map, exp_driver_map = make_expert_maps(exp)

    exp_other_map: Dict[str, bool] = {}
    for _, row in exp.iterrows():
        rid = str(row.get("request_id"))
        driver_mode = str(row.get("driver_mode", ""))
        is_other = to_bool(row.get("is_other")) or driver_mode == "OTHER_DIRECT_DRIVER"
        exp_other_map[rid] = bool(is_other)

    common_ids = [rid for rid in sched["request_id"].astype(str).tolist() if rid in set(exp["request_id"].astype(str))]
    denom_all = len(common_ids) or len(sched)
    vehicle_count = 0
    driver_count = 0
    driver_denom = 0
    other_count = 0
    other_denom = 0
    unserved_count = 0

    for _, row in sched.iterrows():
        rid = str(row.get("request_id"))
        if rid not in exp["request_id"].astype(str).values:
            continue
        assigned = str(row.get("status")) == "ASSIGNED"
        candidate_vehicle = as_key(row.get("assigned_vehicle_id"))
        expert_vehicle = exp_vehicle_map.get(rid)
        if assigned and candidate_vehicle and expert_vehicle and str(candidate_vehicle) == str(expert_vehicle):
            vehicle_count += 1

        candidate_is_other = str(row.get("assignment_mode")) == "OTHER_DIRECT_DRIVER" or to_bool(row.get("is_other"))
        expert_is_other = exp_other_map.get(rid, False)
        if expert_is_other or candidate_is_other:
            other_denom += 1
            if candidate_is_other == expert_is_other:
                other_count += 1
        else:
            driver_denom += 1
            cand_driver_name = row.get("assigned_driver_name")
            expert_driver = exp_driver_map.get(rid)
            if assigned and expert_driver is not None and str(cand_driver_name) == str(expert_driver):
                driver_count += 1

        # The current expert table has no explicit unserved column.  If later
        # expert tables include UNSERVED rows, this will start reflecting them.
        expert_unserved = False
        cand_unserved = str(row.get("status")) == "UNSERVED"
        if cand_unserved == expert_unserved:
            unserved_count += 1

    vehicle_rate = _safe_rate(vehicle_count, denom_all)
    driver_rate = _safe_rate(driver_count, driver_denom)
    other_rate = _safe_rate(other_count, other_denom) if other_denom else 100.0
    unserved_rate = _safe_rate(unserved_count, denom_all)
    expert_score = (
        0.35 * vehicle_rate
        + 0.25 * driver_rate
        + 0.20 * other_rate
        + 0.20 * unserved_rate
    )
    return {
        "vehicle_match_count": int(vehicle_count),
        "vehicle_match_rate": round(vehicle_rate, 3),
        "driver_match_count": int(driver_count),
        "driver_match_rate": round(driver_rate, 3),
        "other_match_count": int(other_count),
        "other_match_rate": round(other_rate, 3),
        "unserved_match_count": int(unserved_count),
        "unserved_match_rate": round(unserved_rate, 3),
        "expert_similarity_score": round(expert_score, 3),
    }


def compute_candidate_feature_row(
    schedule: pd.DataFrame,
    score_summary: Dict[str, Any] | pd.Series | pd.DataFrame,
    expert_schedule: pd.DataFrame,
    scenario_name: str = "baseline",
) -> Dict[str, Any]:
    """Build one candidate-level feature row for Ranker training/scoring."""
    if isinstance(score_summary, pd.DataFrame):
        summary = score_summary.iloc[0].to_dict() if not score_summary.empty else {}
    elif isinstance(score_summary, pd.Series):
        summary = score_summary.to_dict()
    else:
        summary = dict(score_summary or {})

    sched = schedule.copy() if schedule is not None else pd.DataFrame()
    assigned = sched[sched.get("status", pd.Series(dtype=str)).eq("ASSIGNED")].copy() if not sched.empty else pd.DataFrame()
    total_requests = max(1, len(sched))
    unserved_count = int(summary.get("unserved_count", (sched.get("status", pd.Series(dtype=str)) == "UNSERVED").sum() if not sched.empty else 0) or 0)
    assigned_count = int(summary.get("assigned_count", len(assigned)) or 0)
    other_assigned_count = int(summary.get("other_assigned_count", 0) or 0)
    if not assigned.empty and "assignment_mode" in assigned.columns:
        other_assigned_count = int((assigned["assignment_mode"] == "OTHER_DIRECT_DRIVER").sum())

    other_total = int(sched.get("is_other", pd.Series(dtype=bool)).fillna(False).map(to_bool).sum()) if not sched.empty and "is_other" in sched.columns else 0
    other_unserved_count = 0
    if not sched.empty and "is_other" in sched.columns:
        other_unserved_count = int((sched["is_other"].map(to_bool) & sched["status"].eq("UNSERVED")).sum())

    # Weighted unserved priority: higher priority unserved requests should hurt.
    weighted_unserved = 0.0
    if not sched.empty and "priority_initial" in sched.columns:
        uns = sched[sched["status"] == "UNSERVED"]
        weighted_unserved = float(pd.to_numeric(uns["priority_initial"], errors="coerce").fillna(1).sum())

    # Workload proxy metrics.
    driver_minutes = pd.Series(dtype=float)
    if not assigned.empty and "assigned_driver_id" in assigned.columns:
        driver_rows = assigned[assigned["assignment_mode"] == "DRIVER_REQUIRED"].copy()
        if not driver_rows.empty:
            driver_minutes = pd.to_numeric(driver_rows["duration_min"], errors="coerce").fillna(0).groupby(driver_rows["assigned_driver_id"].astype(str)).sum()
    max_driver_minutes = float(driver_minutes.max()) if len(driver_minutes) else 0.0
    mean_driver_minutes = float(driver_minutes.mean()) if len(driver_minutes) else 0.0
    fairness_score = max(0.0, 100.0 - _safe_rate(max_driver_minutes - mean_driver_minutes, max(max_driver_minutes, 1.0)))
    fatigue_score = max(0.0, 100.0 - min(100.0, max_driver_minutes / 12.0))  # 1200 minutes -> 0 proxy

    vehicle_minutes = pd.Series(dtype=float)
    if not assigned.empty and "assigned_vehicle_id" in assigned.columns:
        vehicle_minutes = pd.to_numeric(assigned["duration_min"], errors="coerce").fillna(0).groupby(assigned["assigned_vehicle_id"].astype(str)).sum()
    max_vehicle_minutes = float(vehicle_minutes.max()) if len(vehicle_minutes) else 0.0
    mean_vehicle_minutes = float(vehicle_minutes.mean()) if len(vehicle_minutes) else 0.0
    vehicle_preservation_score = max(0.0, 100.0 - _safe_rate(max_vehicle_minutes - mean_vehicle_minutes, max(max_vehicle_minutes, 1.0)))

    # Other handling score rewards no driver assigned to Other rows and successful Other support.
    other_driver_misassigned = 0
    if not sched.empty and "assignment_mode" in sched.columns:
        other_rows = sched[sched["assignment_mode"].eq("OTHER_DIRECT_DRIVER")]
        if not other_rows.empty and "assigned_driver_id" in other_rows.columns:
            other_driver_misassigned = int(other_rows["assigned_driver_id"].fillna("").astype(str).str.strip().ne("").sum())
    other_handling_score = 100.0
    if other_total:
        other_handling_score = max(0.0, 100.0 - _safe_rate(other_unserved_count + other_driver_misassigned, other_total))

    vehicle_fit_sum = float(summary.get("vehicle_fit_score_sum", assigned.get("vehicle_fit_score", pd.Series(dtype=float)).fillna(0).sum() if not assigned.empty else 0) or 0)
    driver_fit_sum = float(summary.get("driver_fit_score_sum", assigned.get("driver_fit_score", pd.Series(dtype=float)).fillna(0).sum() if not assigned.empty else 0) or 0)
    vehicle_fit_norm = _safe_rate(vehicle_fit_sum, max(1, assigned_count * 100))
    driver_required_count = int((sched.get("assignment_mode", pd.Series(dtype=str)) == "DRIVER_REQUIRED").sum()) if not sched.empty else 0
    driver_fit_norm = _safe_rate(driver_fit_sum, max(1, driver_required_count * 100))

    expert_metrics = compute_expert_similarity_metrics(sched, expert_schedule)

    hard_violation_count = int(summary.get("hard_violation_count", 0) or 0)
    diversity = _safe_num(summary.get("diversity_score_vs_previous", summary.get("diversity_score", 0)))
    support_score = _safe_rate(assigned_count, total_requests)
    weighted_support_score = max(0.0, 100.0 - _safe_rate(weighted_unserved, max(1, total_requests * 3)))

    row = {
        "scenario_name": scenario_name,
        "candidate_id": summary.get("candidate_id", sched.get("candidate_id", pd.Series([""])).iloc[0] if not sched.empty and "candidate_id" in sched.columns else ""),
        "candidate_profile": summary.get("candidate_profile", sched.get("candidate_profile", pd.Series([""])).iloc[0] if not sched.empty and "candidate_profile" in sched.columns else ""),
        "solver_status": summary.get("solver_status", ""),
        "total_requests": int(total_requests),
        "assigned_count": int(assigned_count),
        "unserved_count": int(unserved_count),
        "weighted_unserved_priority": round(weighted_unserved, 3),
        "hard_violation_count": hard_violation_count,
        "vehicle_fit_score_sum": vehicle_fit_sum,
        "driver_fit_score_sum": driver_fit_sum,
        "vehicle_fit_norm": round(vehicle_fit_norm, 3),
        "driver_fit_norm": round(driver_fit_norm, 3),
        "fairness_score": round(fairness_score, 3),
        "fatigue_score": round(fatigue_score, 3),
        "vehicle_preservation_score": round(vehicle_preservation_score, 3),
        "other_assigned_count": int(other_assigned_count),
        "other_total_count": int(other_total),
        "other_unserved_count": int(other_unserved_count),
        "other_driver_misassigned_count": int(other_driver_misassigned),
        "other_handling_score": round(other_handling_score, 3),
        "diversity_score": round(diversity, 3),
        "support_score": round(support_score, 3),
        "weighted_support_score": round(weighted_support_score, 3),
        "max_driver_minutes": round(max_driver_minutes, 3),
        "max_vehicle_minutes": round(max_vehicle_minutes, 3),
    }
    row.update(expert_metrics)
    # Alias names used by the Ranker/inverse-optimization specification.
    row["expert_vehicle_match_rate"] = row.get("vehicle_match_rate", 0.0)
    row["expert_driver_match_rate"] = row.get("driver_match_rate", 0.0)
    row["expert_other_match_rate"] = row.get("other_match_rate", 0.0)
    return row
