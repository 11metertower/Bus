"""Generate artificial shortage scenarios for the G-LIFT MVP.

The user should not have to handcraft unsupported-case Excel files.  This
module copies the standardized CSV files produced by preprocessing and then
modifies availability/skill tables in controlled ways so the Streamlit demo can
show realistic 미지원 cases and populated unserved-reason reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from .config import PROCESSED_DIR, SCENARIOS_DIR, SCENARIO_NAMES, TIME_SLOT_MINUTES

STANDARD_FILES = [
    "requests.csv",
    "vehicles.csv",
    "drivers.csv",
    "vehicle_availability.csv",
    "driver_availability.csv",
    "expert_schedule.csv",
]


@dataclass
class ScenarioResult:
    scenario_name: str
    scenario_dir: Path
    target_request_ids: List[str]
    description: str
    metadata_path: Path


def _read_base_tables(processed_dir: Path = PROCESSED_DIR) -> Dict[str, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    tables: Dict[str, pd.DataFrame] = {}
    for filename in STANDARD_FILES:
        key = filename.replace(".csv", "")
        path = processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing processed CSV: {path}")
        tables[key] = pd.read_csv(path)
    return tables


def _write_scenario_tables(scenario_dir: Path, tables: Dict[str, pd.DataFrame], metadata: Dict) -> ScenarioResult:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for filename in STANDARD_FILES:
        key = filename.replace(".csv", "")
        tables[key].to_csv(scenario_dir / filename, index=False, encoding="utf-8-sig")
    metadata_path = scenario_dir / "scenario_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return ScenarioResult(
        scenario_name=metadata["scenario_name"],
        scenario_dir=scenario_dir,
        target_request_ids=metadata.get("target_request_ids", []),
        description=metadata.get("description", ""),
        metadata_path=metadata_path,
    )


def _slots_for_requests(requests: pd.DataFrame, request_ids: Sequence[str]) -> List[int]:
    slots: set[int] = set()
    reqs = requests[requests["request_id"].astype(str).isin([str(x) for x in request_ids])]
    for _, row in reqs.iterrows():
        start = int(row["start_min"])
        end = int(row["end_min"])
        slots.update(range(start // TIME_SLOT_MINUTES, max(start // TIME_SLOT_MINUTES, (end - 1) // TIME_SLOT_MINUTES + 1)))
    return sorted(slots)


def _matching_vehicle_ids(vehicles: pd.DataFrame, vehicle_classes: Iterable[str]) -> List[str]:
    classes = {str(c) for c in vehicle_classes}
    df = vehicles[vehicles["vehicle_class"].astype(str).isin(classes)].copy()
    # Keep maintenance vehicles in the list too; setting availability false is harmless.
    return df["vehicle_id"].astype(str).tolist()


def _set_vehicle_unavailable(tables: Dict[str, pd.DataFrame], vehicle_ids: Sequence[str], slots: Sequence[int]) -> int:
    va = tables["vehicle_availability"].copy()
    ids = {str(v) for v in vehicle_ids}
    slot_set = {int(s) for s in slots}
    mask = va["resource_id"].astype(str).isin(ids) & va["slot_index"].astype(int).isin(slot_set)
    va.loc[mask, "is_available"] = False
    va.loc[mask, "raw_value"] = "F"
    tables["vehicle_availability"] = va
    return int(mask.sum())


def _set_driver_unavailable(tables: Dict[str, pd.DataFrame], driver_ids: Sequence[str], slots: Sequence[int]) -> int:
    da = tables["driver_availability"].copy()
    ids = {str(d) for d in driver_ids}
    slot_set = {int(s) for s in slots}
    mask = da["driver_id"].astype(str).isin(ids) & da["slot_index"].astype(int).isin(slot_set)
    da.loc[mask, "is_available"] = False
    da.loc[mask, "raw_value"] = "F"
    tables["driver_availability"] = da
    return int(mask.sum())


def _copy_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {k: v.copy(deep=True) for k, v in tables.items()}


def _targets_by_class(requests: pd.DataFrame, vehicle_class: str, n: int = 3, other: bool | None = None) -> List[str]:
    df = requests[requests["required_vehicle_class"].astype(str) == vehicle_class].copy()
    if other is not None:
        df = df[df["driver_mode"].astype(str).eq("OTHER_DIRECT_DRIVER") if other else ~df["driver_mode"].astype(str).eq("OTHER_DIRECT_DRIVER")]
    if df.empty:
        df = requests.copy()
    df = df.sort_values(["priority_initial", "duration_min"], ascending=[False, False])
    return df["request_id"].astype(str).head(n).tolist()


def _target_other_requests(requests: pd.DataFrame, n: int = 2) -> List[str]:
    df = requests[requests["driver_mode"].astype(str).eq("OTHER_DIRECT_DRIVER")].copy()
    if df.empty:
        return []
    return df.sort_values(["duration_min", "priority_initial"], ascending=[False, False])["request_id"].astype(str).head(n).tolist()


def _target_driver_requests(requests: pd.DataFrame, n: int = 4) -> List[str]:
    df = requests[~requests["driver_mode"].astype(str).eq("OTHER_DIRECT_DRIVER")].copy()
    if df.empty:
        return []
    return df.sort_values(["priority_initial", "duration_min"], ascending=[False, False])["request_id"].astype(str).head(n).tolist()


def _choose_skill_slot_for_targets(requests: pd.DataFrame, vehicles: pd.DataFrame, target_ids: Sequence[str]) -> str:
    """Pick a skill column to degrade based on target vehicle classes."""
    targets = requests[requests["request_id"].astype(str).isin([str(x) for x in target_ids])]
    text = " ".join(targets.get("required_vehicle_class", pd.Series(dtype=str)).astype(str).tolist())
    if any(word in text for word in ["대형", "버스", "9톤", "5톤"]):
        return "skill_large"
    if any(word in text for word in ["1톤", "지프"]):
        return "skill_medium"
    return "skill_small"


def _make_metadata(name: str, description: str, target_ids: List[str], manipulations: List[Dict]) -> Dict:
    return {
        "scenario_name": name,
        "description": description,
        "target_request_ids": target_ids,
        "manipulations": manipulations,
        "generated_from": "data/processed/*.csv",
        "note": "가상/익명화된 본선 시연용 부족 시나리오입니다. 원본 데이터는 변경하지 않습니다.",
    }


def generate_all_scenarios(processed_dir: Path = PROCESSED_DIR, scenarios_dir: Path = SCENARIOS_DIR) -> List[ScenarioResult]:
    """Generate baseline and five shortage scenarios."""
    base = _read_base_tables(processed_dir)
    scenarios_dir = Path(scenarios_dir)
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    results: List[ScenarioResult] = []

    # Baseline: exact copy of processed inputs.
    baseline_meta = _make_metadata(
        "baseline",
        "전처리 결과를 그대로 사용하는 정상 시나리오입니다.",
        [],
        [],
    )
    results.append(_write_scenario_tables(scenarios_dir / "baseline", _copy_tables(base), baseline_meta))

    # 1. Vehicle shortage: make all vehicles of a common requested class unavailable for target intervals.
    tables = _copy_tables(base)
    requests = tables["requests"]
    vehicles = tables["vehicles"]
    class_counts = requests["required_vehicle_class"].astype(str).value_counts()
    target_class = class_counts.index[0] if not class_counts.empty else str(requests.iloc[0]["required_vehicle_class"])
    target_ids = _targets_by_class(requests, target_class, n=4)
    slots = _slots_for_requests(requests, target_ids)
    vehicle_ids = _matching_vehicle_ids(vehicles, [target_class])
    changed = _set_vehicle_unavailable(tables, vehicle_ids, slots)
    meta = _make_metadata(
        "vehicle_shortage",
        f"{target_class} 차량을 목표 배차 시간대에 강제로 사용불가 처리한 차량 부족 시나리오입니다.",
        target_ids,
        [{"action": "set_vehicle_availability_false", "vehicle_class": target_class, "vehicle_ids": vehicle_ids, "slot_count": len(slots), "changed_cells": changed}],
    )
    results.append(_write_scenario_tables(scenarios_dir / "vehicle_shortage", tables, meta))

    # 2. Driver shortage: make every driver unavailable during high-priority driver-required requests.
    tables = _copy_tables(base)
    requests = tables["requests"]
    target_ids = _target_driver_requests(requests, n=5)
    slots = _slots_for_requests(requests, target_ids)
    driver_ids = tables["drivers"]["driver_id"].astype(str).tolist()
    changed = _set_driver_unavailable(tables, driver_ids, slots)
    meta = _make_metadata(
        "driver_shortage",
        "운전병 지원 배차 시간대에 전체 운전병 가용성을 일부 강제로 False 처리한 운전병 부족 시나리오입니다.",
        target_ids,
        [{"action": "set_driver_availability_false", "driver_count": len(driver_ids), "slot_count": len(slots), "changed_cells": changed}],
    )
    results.append(_write_scenario_tables(scenarios_dir / "driver_shortage", tables, meta))

    # 3. Skill shortage: degrade the relevant skill slot for all drivers.
    tables = _copy_tables(base)
    requests = tables["requests"]
    candidates = requests[~requests["driver_mode"].astype(str).eq("OTHER_DIRECT_DRIVER")].copy()
    candidates = candidates[candidates["request_type"].astype(str).str.contains("장거리|단거리", na=False)]
    if candidates.empty:
        candidates = requests[~requests["driver_mode"].astype(str).eq("OTHER_DIRECT_DRIVER")].copy()
    target_ids = candidates.sort_values(["priority_initial", "duration_min"], ascending=[False, False])["request_id"].astype(str).head(4).tolist()
    skill_slot = _choose_skill_slot_for_targets(requests, tables["vehicles"], target_ids)
    old_values = tables["drivers"][skill_slot].astype(str).value_counts().to_dict()
    tables["drivers"].loc[:, skill_slot] = 4
    tables["drivers"].loc[:, "skill_valid"] = True
    meta = _make_metadata(
        "skill_shortage",
        f"목표 배차에 필요한 {skill_slot} 기량을 전체 운전병에 대해 운전불가 수준으로 낮춘 기량 부족 시나리오입니다.",
        target_ids,
        [{"action": "degrade_driver_skill", "skill_slot": skill_slot, "old_value_counts": old_values, "new_value": 4}],
    )
    results.append(_write_scenario_tables(scenarios_dir / "skill_shortage", tables, meta))

    # 4. Other vehicle conflict: make compatible vehicles unavailable during Other request intervals.
    tables = _copy_tables(base)
    requests = tables["requests"]
    other_ids = _target_other_requests(requests, n=3)
    target_classes = requests[requests["request_id"].astype(str).isin(other_ids)]["required_vehicle_class"].astype(str).unique().tolist()
    slots = _slots_for_requests(requests, other_ids)
    vehicle_ids = _matching_vehicle_ids(tables["vehicles"], target_classes)
    changed = _set_vehicle_unavailable(tables, vehicle_ids, slots)
    meta = _make_metadata(
        "other_vehicle_conflict",
        "Other 직접운전 배차는 운전병이 필요 없지만 차량 자원이 부족하여 미지원되는 상황을 보여주는 시나리오입니다.",
        other_ids,
        [{"action": "set_other_compatible_vehicles_unavailable", "vehicle_classes": target_classes, "vehicle_ids": vehicle_ids, "slot_count": len(slots), "changed_cells": changed}],
    )
    results.append(_write_scenario_tables(scenarios_dir / "other_vehicle_conflict", tables, meta))

    # 5. Mixed shortage: combine vehicle, driver, skill, and Other vehicle shortage.
    tables = _copy_tables(base)
    requests = tables["requests"]
    vehicles = tables["vehicles"]
    mixed_targets: List[str] = []
    manipulations: List[Dict] = []
    vehicle_ids_all: List[str] = []

    # vehicle component on the most common class
    target_class = requests["required_vehicle_class"].astype(str).value_counts().index[0]
    ids = _targets_by_class(requests, target_class, n=3)
    mixed_targets.extend(ids)
    slots = _slots_for_requests(requests, ids)
    v_ids = _matching_vehicle_ids(vehicles, [target_class])
    changed = _set_vehicle_unavailable(tables, v_ids, slots)
    vehicle_ids_all.extend(v_ids)
    manipulations.append({"action": "mixed_vehicle_shortage", "vehicle_class": target_class, "target_request_ids": ids, "changed_cells": changed})

    # driver component
    ids = _target_driver_requests(requests, n=3)
    mixed_targets.extend(ids)
    slots = _slots_for_requests(requests, ids)
    d_ids = tables["drivers"]["driver_id"].astype(str).tolist()
    changed = _set_driver_unavailable(tables, d_ids, slots)
    manipulations.append({"action": "mixed_driver_shortage", "target_request_ids": ids, "changed_cells": changed})

    # skill component
    skill_slot = _choose_skill_slot_for_targets(requests, vehicles, ids)
    tables["drivers"].loc[:, skill_slot] = 4
    tables["drivers"].loc[:, "skill_valid"] = True
    manipulations.append({"action": "mixed_skill_shortage", "skill_slot": skill_slot, "new_value": 4})

    # Other component
    other_ids = _target_other_requests(requests, n=2)
    mixed_targets.extend(other_ids)
    other_classes = requests[requests["request_id"].astype(str).isin(other_ids)]["required_vehicle_class"].astype(str).unique().tolist()
    slots = _slots_for_requests(requests, other_ids)
    v_ids = _matching_vehicle_ids(vehicles, other_classes)
    changed = _set_vehicle_unavailable(tables, v_ids, slots)
    manipulations.append({"action": "mixed_other_vehicle_conflict", "target_request_ids": other_ids, "vehicle_classes": other_classes, "changed_cells": changed})

    mixed_targets = sorted(set([str(x) for x in mixed_targets]))
    meta = _make_metadata(
        "mixed_shortage",
        "차량 부족, 운전병 부족, 기량 부족, Other 차량 충돌을 함께 반영한 본선 시연용 복합 부족 시나리오입니다.",
        mixed_targets,
        manipulations,
    )
    results.append(_write_scenario_tables(scenarios_dir / "mixed_shortage", tables, meta))

    summary = pd.DataFrame([r.__dict__ | {"scenario_dir": str(r.scenario_dir), "metadata_path": str(r.metadata_path)} for r in results])
    summary.to_csv(scenarios_dir / "scenario_generation_summary.csv", index=False, encoding="utf-8-sig")
    return results


def main() -> None:
    results = generate_all_scenarios(PROCESSED_DIR, SCENARIOS_DIR)
    print("G-LIFT shortage scenarios generated.")
    for r in results:
        print(f"- {r.scenario_name}: {r.scenario_dir} / targets={r.target_request_ids}")


if __name__ == "__main__":
    main()
