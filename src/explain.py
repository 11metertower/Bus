"""Explanation helpers for G-LIFT MVP optimizer outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .feature_extractor import (
    as_key,
    driver_has_required_skill,
    is_resource_available,
    request_slots,
    vehicle_matches_request,
)


def assignment_explanation(row: pd.Series) -> str:
    """Return a concise human-readable explanation for one schedule row."""
    if row.get("status") == "UNSERVED":
        detail = row.get("unserved_reason_detail") or "지원 가능한 차량/운전병 조합이 부족하여 미지원 처리되었습니다."
        return str(detail)
    if row.get("assignment_mode") == "OTHER_DIRECT_DRIVER":
        return (
            "Other 직접운전 배차이므로 운전병을 배정하지 않고 차량만 배정했습니다. "
            "차량 가용성, 시간 중복, 차종 조건은 그대로 적용했습니다."
        )
    return (
        "운전병 지원 배차로 차량과 운전병을 함께 배정했습니다. "
        "차량 시간 중복, 운전병 시간 중복, 차종, 기량, 가용성 조건을 만족합니다."
    )


def diagnose_unserved_reason(
    request: pd.Series,
    vehicles: pd.DataFrame,
    drivers: pd.DataFrame,
    vehicle_availability_index: Dict[str, set],
    driver_availability_index: Dict[str, set],
) -> Dict[str, str]:
    """Diagnose why a request may have remained unserved.

    This is a deterministic explanation helper, not a proof of infeasibility.
    It checks the main bottleneck layers in a transparent order.
    """
    slots = request_slots(request.get("start_min"), request.get("end_min"))
    is_other = bool(request.get("driver_mode") == "OTHER_DIRECT_DRIVER" or request.get("requires_driver") is False)

    compatible_vehicles = vehicles[vehicles.apply(lambda v: vehicle_matches_request(v, request), axis=1)].copy()
    if compatible_vehicles.empty:
        return {
            "reason_code": "NO_COMPATIBLE_VEHICLE_CLASS",
            "reason_detail": f"요청 차종({request.get('required_vehicle_class')})과 일치하는 차량이 없습니다.",
            "possible_relaxation": "요청 차종을 재확인하거나 대체 가능 차종 허용 여부를 지휘관이 판단해야 합니다.",
        }

    usable_vehicles = compatible_vehicles[~compatible_vehicles["is_maintenance_or_unavailable"].fillna(False).astype(bool)]
    if usable_vehicles.empty:
        return {
            "reason_code": "ALL_COMPATIBLE_VEHICLES_UNAVAILABLE_MAINTENANCE",
            "reason_detail": "요청 차종 차량이 모두 정비/입고/사용불가 상태입니다.",
            "possible_relaxation": "정비 상태를 재확인하거나 대체 차종 지원 가능성을 검토하십시오.",
        }

    time_ok_vehicles = [
        v for _, v in usable_vehicles.iterrows()
        if is_resource_available(v.get("vehicle_id"), slots, vehicle_availability_index)
    ]
    if not time_ok_vehicles:
        if is_other:
            return {
                "reason_code": "OTHER_VEHICLE_RESOURCE_CONFLICT",
                "reason_detail": "Other 직접운전 배차라 운전병은 필요 없지만, 요청 시간대에 사용할 수 있는 해당 차종 차량이 없습니다.",
                "possible_relaxation": "동일 차종 차량 추가 가용 여부를 확인하거나 Other 배차의 실제 사용시간을 재확인하십시오.",
            }
        return {
            "reason_code": "NO_VEHICLE_AVAILABLE_IN_TIME_WINDOW",
            "reason_detail": "요청 시간대에 해당 차종 차량이 모두 다른 일정 또는 사용불가 시간과 충돌합니다.",
            "possible_relaxation": "사용부서에 실제 사용시간을 재확인하거나 하위 우선순위 배차 조정을 검토하십시오.",
        }

    if is_other:
        return {
            "reason_code": "OTHER_VEHICLE_RESOURCE_CONFLICT",
            "reason_detail": "Other 직접운전 배차라 운전병은 필요 없지만, 전역 최적화 과정에서 같은 차량 자원을 더 높은 우선순위 배차에 배정했습니다.",
            "possible_relaxation": "동일 차종 차량 추가 가용 여부 또는 해당 배차의 시간 조정을 확인하십시오.",
        }

    skill_ok_pairs = []
    for v in time_ok_vehicles:
        for _, d in drivers.iterrows():
            if driver_has_required_skill(d, v, request):
                skill_ok_pairs.append((v, d))
    if not skill_ok_pairs:
        return {
            "reason_code": "NO_DRIVER_WITH_REQUIRED_SKILL",
            "reason_detail": "차량 후보는 있으나 요구 차종/운행범위 기량을 만족하는 운전병이 없습니다.",
            "possible_relaxation": "운전병 기량 현황을 재확인하거나 간부 직접운전/Other 전환 가능성을 검토하십시오.",
        }

    available_pairs = [
        (v, d) for v, d in skill_ok_pairs
        if is_resource_available(d.get("driver_id"), slots, driver_availability_index)
    ]
    if not available_pairs:
        return {
            "reason_code": "NO_DRIVER_AVAILABLE_IN_TIME_WINDOW",
            "reason_detail": "요구 기량 운전병은 있으나 요청 시간대에는 모두 휴가, 교육, 기존 일정 등으로 가용하지 않습니다.",
            "possible_relaxation": "운전병 스케줄 또는 배차 시간을 재조정하십시오.",
        }

    return {
        "reason_code": "LOWER_PRIORITY_THAN_CONFLICTING_REQUESTS",
        "reason_detail": "가능 후보는 있었으나 같은 차량/운전병 자원을 공유하는 더 높은 우선순위 배차를 살리기 위해 미지원 처리되었습니다.",
        "possible_relaxation": "지휘관 판단으로 우선순위를 변경하거나 상충 배차의 시간 조정을 검토하십시오.",
    }
