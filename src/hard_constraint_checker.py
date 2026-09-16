"""Hard-constraint verification for generated G-LIFT schedules.

This module is intentionally separate from the CP-SAT model.  The solver
*should* create only legal schedules, but a public MVP needs an independent
checker that can be shown to judges: vehicle overlap = 0, driver overlap = 0,
Other driver assignment errors = 0, and so on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from .feature_extractor import (
    as_key,
    build_availability_index,
    driver_has_required_skill,
    is_resource_available,
    request_slots,
)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "t", "yes", "y"}


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """End-exclusive interval overlap: [start, end)."""
    return int(a_start) < int(b_end) and int(b_start) < int(a_end)


def _summary_row(check_name: str, issue_type: str, count: int, message: str, examples: List[str]) -> Dict[str, Any]:
    return {
        "check_name": check_name,
        "issue_type": issue_type,
        "severity": "ERROR" if count else "OK",
        "violation_count": int(count),
        "examples": "; ".join(examples[:10]),
        "message": message if count else f"{check_name}: 위반 없음",
    }


def run_hard_constraint_checks(schedule: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a compact hard-constraint check report.

    Output is one row per check type with a violation count.  This is easier to
    present in Streamlit and Excel than a huge log, and it also gives the MVP a
    clear success criterion: every violation_count should be 0.
    """
    rows: List[Dict[str, Any]] = []
    if schedule is None or schedule.empty:
        rows.append(_summary_row("SCHEDULE_NOT_EMPTY", "EMPTY_SCHEDULE", 1, "배차표가 비어 있습니다.", ["no schedule rows"]))
        return pd.DataFrame(rows)

    vehicles = tables.get("vehicles", pd.DataFrame()).copy()
    drivers = tables.get("drivers", pd.DataFrame()).copy()
    requests = tables.get("requests", pd.DataFrame()).copy()

    if not vehicles.empty:
        vehicles["vehicle_id"] = vehicles["vehicle_id"].map(as_key)
    if not drivers.empty:
        drivers["driver_id"] = drivers["driver_id"].astype(str)
    if not requests.empty:
        requests["request_id"] = requests["request_id"].astype(str)

    vehicle_by_id = {as_key(v.get("vehicle_id")): v for _, v in vehicles.iterrows()}
    driver_by_id = {str(d.get("driver_id")): d for _, d in drivers.iterrows()}
    request_by_id = {str(r.get("request_id")): r for _, r in requests.iterrows()}

    vehicle_availability_index = build_availability_index(tables.get("vehicle_availability", pd.DataFrame()), "resource_id")
    driver_availability_index = build_availability_index(tables.get("driver_availability", pd.DataFrame()), "driver_id")

    assigned = schedule[schedule.get("status", "") == "ASSIGNED"].copy()

    # 1. one row per request / no duplicate assigned rows
    dup_examples: List[str] = []
    violation_count = 0
    if "request_id" in schedule.columns:
        counts = schedule["request_id"].astype(str).value_counts()
        bad = counts[counts != 1]
        violation_count = int(len(bad))
        dup_examples = [f"{rid}: rows={cnt}" for rid, cnt in bad.items()]
    rows.append(_summary_row("배차별 단일 행", "REQUEST_ROW_COUNT_NOT_ONE", violation_count, "한 요청이 결과표에 0개 또는 2개 이상 존재합니다.", dup_examples))

    # 2. Other driver misassignment
    examples = []
    if not assigned.empty:
        bad_other = assigned[(assigned["assignment_mode"] == "OTHER_DIRECT_DRIVER") & (assigned["assigned_driver_id"].fillna("").astype(str).str.strip() != "")]
        examples = bad_other["request_id"].astype(str).tolist()
        rows.append(_summary_row("Other 운전병 미배정", "OTHER_DRIVER_MISASSIGNED", len(examples), "Other 직접운전 배차에 운전병이 배정되었습니다.", examples))
    else:
        rows.append(_summary_row("Other 운전병 미배정", "OTHER_DRIVER_MISASSIGNED", 0, "", []))

    # 3. vehicle class / maintenance / availability / skill / driver availability
    class_examples, maint_examples, veh_avail_examples = [], [], []
    skill_examples, drv_avail_examples, unknown_vehicle_examples, unknown_driver_examples = [], [], [], []
    for _, row in assigned.iterrows():
        rid = str(row.get("request_id"))
        vid = as_key(row.get("assigned_vehicle_id"))
        vehicle = vehicle_by_id.get(vid)
        slots = request_slots(row.get("start_min"), row.get("end_min"))
        if vehicle is None:
            unknown_vehicle_examples.append(f"{rid}: {vid}")
            continue
        if str(vehicle.get("vehicle_class")) != str(row.get("required_vehicle_class")):
            class_examples.append(f"{rid}: req={row.get('required_vehicle_class')}, veh={vehicle.get('vehicle_class')}")
        if _bool(vehicle.get("is_maintenance_or_unavailable")):
            maint_examples.append(f"{rid}: {vid}")
        if not is_resource_available(vid, slots, vehicle_availability_index):
            veh_avail_examples.append(f"{rid}: {vid}")

        if row.get("assignment_mode") == "DRIVER_REQUIRED":
            did = str(row.get("assigned_driver_id"))
            driver = driver_by_id.get(did)
            if driver is None:
                unknown_driver_examples.append(f"{rid}: {did}")
            else:
                request_like = request_by_id.get(rid, pd.Series({"request_type": row.get("request_type")}))
                if not driver_has_required_skill(driver, vehicle, request_like):
                    skill_examples.append(f"{rid}: {did}")
                if not is_resource_available(did, slots, driver_availability_index):
                    drv_avail_examples.append(f"{rid}: {did}")

    rows.append(_summary_row("배정 차량 존재", "UNKNOWN_VEHICLE", len(unknown_vehicle_examples), "배정된 차량이 차량 목록에 없습니다.", unknown_vehicle_examples))
    rows.append(_summary_row("차종 일치", "VEHICLE_CLASS_MISMATCH", len(class_examples), "요청 차종과 배정 차량 차종이 다릅니다.", class_examples))
    rows.append(_summary_row("정비/입고 차량 제외", "MAINTENANCE_VEHICLE_ASSIGNED", len(maint_examples), "정비/입고/사용불가 차량이 배정되었습니다.", maint_examples))
    rows.append(_summary_row("차량 가용성", "VEHICLE_NOT_AVAILABLE", len(veh_avail_examples), "배정 차량이 요청 시간대에 전부 가용하지 않습니다.", veh_avail_examples))
    rows.append(_summary_row("배정 운전병 존재", "UNKNOWN_DRIVER", len(unknown_driver_examples), "배정 운전병이 운전병 목록에 없습니다.", unknown_driver_examples))
    rows.append(_summary_row("운전병 기량", "DRIVER_SKILL_MISMATCH", len(skill_examples), "운전병 기량이 요청/차량 조건을 만족하지 않습니다.", skill_examples))
    rows.append(_summary_row("운전병 가용성", "DRIVER_NOT_AVAILABLE", len(drv_avail_examples), "운전병이 요청 시간대에 전부 가용하지 않습니다.", drv_avail_examples))

    # 4. Vehicle overlaps
    veh_overlap_examples = []
    if not assigned.empty:
        for vehicle_id, group in assigned.groupby("assigned_vehicle_id"):
            if vehicle_id in ["", None] or pd.isna(vehicle_id):
                continue
            items = group.sort_values("start_min").to_dict("records")
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    if _overlap(int(a["start_min"]), int(a["end_min"]), int(b["start_min"]), int(b["end_min"])):
                        veh_overlap_examples.append(f"{vehicle_id}: {a['request_id']} & {b['request_id']}")
    rows.append(_summary_row("차량 시간 중복", "VEHICLE_TIME_OVERLAP", len(veh_overlap_examples), "동일 차량이 겹치는 시간대에 중복 배정되었습니다.", veh_overlap_examples))

    # 5. Driver overlaps.  Other is excluded by design.
    drv_overlap_examples = []
    driver_assigned = assigned[assigned.get("assignment_mode", "") == "DRIVER_REQUIRED"]
    if not driver_assigned.empty:
        for driver_id, group in driver_assigned.groupby("assigned_driver_id"):
            if driver_id in ["", None] or pd.isna(driver_id):
                continue
            items = group.sort_values("start_min").to_dict("records")
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    if _overlap(int(a["start_min"]), int(a["end_min"]), int(b["start_min"]), int(b["end_min"])):
                        drv_overlap_examples.append(f"{driver_id}: {a['request_id']} & {b['request_id']}")
    rows.append(_summary_row("운전병 시간 중복", "DRIVER_TIME_OVERLAP", len(drv_overlap_examples), "동일 운전병이 겹치는 시간대에 중복 배정되었습니다.", drv_overlap_examples))

    report = pd.DataFrame(rows)
    report["violation_count"] = report["violation_count"].astype(int)
    return report


def hard_violation_total(report: pd.DataFrame) -> int:
    """Total hard-constraint violations from a check report."""
    if report is None or report.empty:
        return 0
    if "violation_count" in report.columns:
        return int(report.loc[report["severity"] == "ERROR", "violation_count"].fillna(0).sum())
    return int((report.get("severity", pd.Series(dtype=str)) == "ERROR").sum())
