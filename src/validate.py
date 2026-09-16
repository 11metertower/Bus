"""Validation rules for G-LIFT standardized tables."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _add_issue(issues: List[dict], table: str, severity: str, issue_type: str, message: str, row_id=None, column=None, value=None) -> None:
    issues.append({
        "table": table,
        "severity": severity,
        "issue_type": issue_type,
        "row_id": row_id,
        "column": column,
        "value": value,
        "message": message,
    })


def validate_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    issues: List[dict] = []

    requests = tables.get("requests", pd.DataFrame())
    vehicles = tables.get("vehicles", pd.DataFrame())
    drivers = tables.get("drivers", pd.DataFrame())
    expert = tables.get("expert_schedule", pd.DataFrame())

    vehicle_ids = set(vehicles.get("vehicle_id", pd.Series(dtype=str)).dropna().astype(str))
    vehicle_classes = set(vehicles.get("vehicle_class", pd.Series(dtype=str)).dropna().astype(str))
    driver_names = set(drivers.get("driver_name", pd.Series(dtype=str)).dropna().astype(str))

    # Request checks
    for _, row in requests.iterrows():
        rid = row.get("request_id")
        for col in ["request_type", "required_vehicle_class", "destination", "requesting_unit", "purpose", "start_min", "end_min"]:
            if pd.isna(row.get(col)) or row.get(col) is None:
                _add_issue(issues, "requests", "ERROR", "missing_required_value", f"필수값 {col} 누락", rid, col, row.get(col))
        if pd.notna(row.get("start_min")) and pd.notna(row.get("end_min")):
            if row.get("end_min") <= row.get("start_min"):
                _add_issue(issues, "requests", "ERROR", "invalid_time_range", "복귀 시각이 출발 시각보다 빠르거나 같습니다.", rid, "start_min/end_min", f"{row.get('start_min')}->{row.get('end_min')}")
        if row.get("required_vehicle_class") not in vehicle_classes:
            _add_issue(issues, "requests", "WARNING", "unknown_vehicle_class", "요청 차종이 차량 목록에 없습니다.", rid, "required_vehicle_class", row.get("required_vehicle_class"))
        if row.get("requested_vehicle_id") and row.get("requested_vehicle_id") not in vehicle_ids:
            _add_issue(issues, "requests", "WARNING", "unknown_requested_vehicle", "요청 차량번호가 차량 목록에 없습니다.", rid, "requested_vehicle_id", row.get("requested_vehicle_id"))
        if row.get("is_other") is True and row.get("requires_driver") is True:
            _add_issue(issues, "requests", "ERROR", "other_driver_mode_conflict", "Other 배차는 requires_driver=False 여야 합니다.", rid, "driver_mode", row.get("driver_mode"))

    # Vehicle checks
    dup_veh = vehicles[vehicles.duplicated("vehicle_id", keep=False)] if "vehicle_id" in vehicles.columns else pd.DataFrame()
    for _, row in dup_veh.iterrows():
        _add_issue(issues, "vehicles", "ERROR", "duplicate_vehicle_id", "차량번호가 중복됩니다.", row.get("vehicle_id"), "vehicle_id", row.get("vehicle_id"))
    for _, row in vehicles.iterrows():
        if not row.get("vehicle_id"):
            _add_issue(issues, "vehicles", "ERROR", "missing_vehicle_id", "차량 번호가 비어 있습니다.", None, "vehicle_id", row.get("vehicle_id"))
        if not row.get("vehicle_class"):
            _add_issue(issues, "vehicles", "ERROR", "missing_vehicle_class", "차종이 비어 있습니다.", row.get("vehicle_id"), "vehicle_class", row.get("vehicle_class"))

    # Driver checks
    dup_drv = drivers[drivers.duplicated("driver_name", keep=False)] if "driver_name" in drivers.columns else pd.DataFrame()
    for _, row in dup_drv.iterrows():
        _add_issue(issues, "drivers", "ERROR", "duplicate_driver_name", "운전병 이름이 중복됩니다.", row.get("driver_id"), "driver_name", row.get("driver_name"))
    for _, row in drivers.iterrows():
        if not row.get("driver_name"):
            _add_issue(issues, "drivers", "ERROR", "missing_driver_name", "운전병 이름이 비어 있습니다.", row.get("driver_id"), "driver_name", row.get("driver_name"))
        if row.get("skill_valid") is not True:
            _add_issue(issues, "drivers", "WARNING", "invalid_or_suspicious_skill", "기량 코드가 1~4 범위의 3자리 코드가 아닐 수 있습니다.", row.get("driver_id"), "skill_code", row.get("skill_code"))
        if row.get("skill_warning"):
            _add_issue(issues, "drivers", "INFO", "skill_parse_warning", str(row.get("skill_warning")), row.get("driver_id"), "skill_raw", row.get("skill_raw"))

    # Expert schedule checks
    for _, row in expert.iterrows():
        rid = row.get("request_id")
        if row.get("assigned_vehicle_id") not in vehicle_ids:
            _add_issue(issues, "expert_schedule", "ERROR", "unknown_assigned_vehicle", "전문가 배차표의 배정 차량이 차량 목록에 없습니다.", rid, "assigned_vehicle_id", row.get("assigned_vehicle_id"))
        assigned_driver = row.get("assigned_driver_raw")
        if row.get("is_other") is True:
            # Correct: Other is not expected to be in driver list.
            pass
        else:
            if not assigned_driver:
                _add_issue(issues, "expert_schedule", "ERROR", "missing_assigned_driver", "운전병 지원 배차인데 전문가 배차표의 운전자가 비어 있습니다.", rid, "assigned_driver_raw", assigned_driver)
            elif assigned_driver not in driver_names:
                _add_issue(issues, "expert_schedule", "WARNING", "assigned_driver_not_in_driver_list", "전문가 배차표의 운전자가 운전자 목록에 없습니다.", rid, "assigned_driver_raw", assigned_driver)

    # Availability checks
    for table_name in ["vehicle_availability", "driver_availability", "vehicle_schedule_assigned_long", "driver_schedule_assigned_long"]:
        df = tables.get(table_name, pd.DataFrame())
        if df.empty:
            _add_issue(issues, table_name, "ERROR", "empty_availability_table", "가용성 테이블이 비어 있습니다.")
            continue
        null_avail = df[df["is_available"].isna()].head(50)
        for _, row in null_avail.iterrows():
            _add_issue(issues, table_name, "WARNING", "unknown_availability_value", "가용성 값이 T/F로 해석되지 않습니다.", row.get("resource_id"), "raw_value", row.get("raw_value"))

    if not issues:
        _add_issue(issues, "all", "OK", "no_issue", "전처리 검증에서 치명적인 오류를 발견하지 못했습니다.")
    return pd.DataFrame(issues)
