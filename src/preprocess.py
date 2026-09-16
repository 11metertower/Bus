"""Data preprocessing pipeline for the G-LIFT MVP.

Run from the project root:

    python -m src.preprocess

Outputs standardized CSV files under data/processed and a validation report
under reports/validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR, STANDARD_TABLE_FILES, VALIDATION_REPORT_DIR
from .data_loader import load_raw_inputs
from .other_handler import add_driver_mode_columns
from .utils import (
    availability_to_bool,
    clean_text,
    default_priority_from_request_type,
    make_driver_id,
    make_request_id,
    minutes_to_hhmm,
    normalize_transmission,
    normalize_vehicle_class,
    normalize_vehicle_number,
    parse_skill_code,
    parse_time_to_minutes,
)
from .validate import validate_tables


def preprocess_requests(raw: pd.DataFrame, *, source: str = "requests_raw") -> pd.DataFrame:
    df = raw.copy()
    required = ["배차 종류", "차량 번호", "차종", "운전자", "행선지", "사용처", "운행 목적", "출발 시각", "복귀 시각"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns: {missing}")

    out = pd.DataFrame()
    out["request_id"] = [make_request_id(i + 1) for i in range(len(df))]
    out["request_type"] = df["배차 종류"].apply(clean_text)
    out["requested_vehicle_id"] = df["차량 번호"].apply(normalize_vehicle_number)
    out["required_vehicle_class"] = df["차종"].apply(normalize_vehicle_class)
    out["driver_raw"] = df["운전자"].apply(clean_text)
    out["destination"] = df["행선지"].apply(clean_text)
    out["requesting_unit"] = df["사용처"].apply(clean_text)
    out["purpose"] = df["운행 목적"].apply(clean_text)
    out["start_min"] = df["출발 시각"].apply(parse_time_to_minutes)
    out["end_min"] = df["복귀 시각"].apply(parse_time_to_minutes)
    out["start_time"] = out["start_min"].apply(minutes_to_hhmm)
    out["end_time"] = out["end_min"].apply(minutes_to_hhmm)
    out["duration_min"] = out["end_min"] - out["start_min"]
    out.loc[out["duration_min"] < 0, "duration_min"] = None
    out["priority_initial"] = out["request_type"].apply(default_priority_from_request_type)
    out["source_table"] = source
    out = add_driver_mode_columns(out, "driver_raw")
    return out


def preprocess_expert_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    out = preprocess_requests(raw, source="expert_best_schedule")
    out = out.rename(columns={
        "requested_vehicle_id": "assigned_vehicle_id",
        "driver_raw": "assigned_driver_raw",
    })
    # Recompute direct-driver columns against the renamed driver column.
    out = out.drop(columns=["is_other", "requires_driver", "driver_mode"], errors="ignore")
    out = add_driver_mode_columns(out, "assigned_driver_raw")
    out["expert_label"] = True
    return out


def preprocess_vehicles(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    required = ["차종", "차량 번호", "자동/수동", "운전 가능 기량", "특이사항"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"vehicles: missing columns: {missing}")
    out = pd.DataFrame()
    out["vehicle_id"] = df["차량 번호"].apply(normalize_vehicle_number)
    out["vehicle_number"] = out["vehicle_id"]
    out["vehicle_class"] = df["차종"].apply(normalize_vehicle_class)
    out["vehicle_class_raw"] = df["차종"].apply(clean_text)
    out["transmission"] = df["자동/수동"].apply(normalize_transmission)
    out["required_skill_text"] = df["운전 가능 기량"].apply(clean_text)
    out["notes"] = df["특이사항"].apply(clean_text)
    note_text = out["notes"].fillna("")
    out["is_rental"] = note_text.str.contains("대여", na=False)
    out["is_electric"] = note_text.str.contains("전기", na=False)
    out["is_maintenance_or_unavailable"] = note_text.str.contains("정비|입고|불가|고장", na=False)
    return out


def preprocess_drivers(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    required = ["이름", "현재 기량 등급"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"drivers: missing columns: {missing}")
    skill_rows = df["현재 기량 등급"].apply(parse_skill_code).apply(pd.Series)
    out = pd.DataFrame()
    out["driver_id"] = [make_driver_id(i + 1) for i in range(len(df))]
    out["driver_name"] = df["이름"].apply(clean_text)
    out = pd.concat([out, skill_rows], axis=1)
    return out


def _detect_metadata_columns(df: pd.DataFrame, resource_type: str) -> list[str]:
    if resource_type == "vehicle":
        return ["차종", "차량 번호", "자동/수동", "운전 가능 기량", "특이사항"]
    if resource_type == "driver":
        return ["이름", "현재 기량 등급", "스케줄"]
    raise ValueError(f"unknown resource_type={resource_type}")


def preprocess_availability_wide(raw: pd.DataFrame, *, resource_type: str, source: str) -> pd.DataFrame:
    """Convert wide T/F availability grid to long format.

    Current sheets use 5-minute slots. The first all-empty time column is skipped,
    leaving 288 slots from 00:00-00:05 through 23:55-24:00.
    """
    df = raw.copy()
    meta_cols = _detect_metadata_columns(df, resource_type)
    missing = [c for c in meta_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing metadata columns: {missing}")

    time_cols = [c for c in df.columns if c not in meta_cols]
    # Drop columns that are entirely blank, such as the first visual 00:00 spacer.
    nonempty_time_cols = [c for c in time_cols if not df[c].isna().all()]

    rows = []
    for row_idx, row in df.iterrows():
        if resource_type == "vehicle":
            resource_id = normalize_vehicle_number(row.get("차량 번호"))
            resource_name = resource_id
            vehicle_class = normalize_vehicle_class(row.get("차종"))
            transmission = normalize_transmission(row.get("자동/수동"))
            schedule_note = clean_text(row.get("특이사항"))
            skill_text = clean_text(row.get("운전 가능 기량"))
        else:
            resource_id = None
            resource_name = clean_text(row.get("이름"))
            vehicle_class = None
            transmission = None
            schedule_note = clean_text(row.get("스케줄"))
            skill_text = clean_text(row.get("현재 기량 등급"))
        if resource_name is None:
            continue
        for slot_idx, col in enumerate(nonempty_time_cols):
            slot_start_min = slot_idx * 5
            slot_end_min = slot_start_min + 5
            raw_value = row.get(col)
            is_available = availability_to_bool(raw_value)
            rows.append({
                "source_table": source,
                "resource_type": resource_type,
                "resource_id": resource_id if resource_type == "vehicle" else resource_name,
                "resource_name": resource_name,
                "vehicle_class": vehicle_class,
                "transmission": transmission,
                "skill_or_requirement": skill_text,
                "schedule_note": schedule_note,
                "slot_index": slot_idx,
                "slot_start_min": slot_start_min,
                "slot_end_min": slot_end_min,
                "slot_start": minutes_to_hhmm(slot_start_min),
                "slot_end": minutes_to_hhmm(slot_end_min),
                "raw_value": clean_text(raw_value),
                "is_available": is_available,
            })
    return pd.DataFrame(rows)


def _attach_driver_ids(availability: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """Attach stable driver_id to driver availability rows by driver name."""
    if availability.empty or drivers.empty:
        return availability
    mapping = drivers[["driver_id", "driver_name"]].dropna().drop_duplicates("driver_name")
    out = availability.merge(mapping, left_on="resource_name", right_on="driver_name", how="left")
    # Keep resource_id human-readable for now, but add stable driver_id for joins.
    cols = list(out.columns)
    if "driver_id" in cols:
        preferred = [c for c in cols if c not in {"driver_id", "driver_name"}]
        insert_at = preferred.index("resource_name") + 1 if "resource_name" in preferred else 0
        preferred.insert(insert_at, "driver_id")
        out = out[preferred]
    return out


def build_processed_tables(raw_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    requests = preprocess_requests(raw_tables["requests_raw"])
    vehicles = preprocess_vehicles(raw_tables["vehicles"])
    drivers = preprocess_drivers(raw_tables["drivers"])
    vehicle_availability = preprocess_availability_wide(raw_tables["vehicle_schedule"], resource_type="vehicle", source="vehicle_schedule")
    driver_availability = preprocess_availability_wide(raw_tables["driver_availability"], resource_type="driver", source="driver_availability")
    vehicle_schedule_assigned_long = preprocess_availability_wide(raw_tables["vehicle_schedule_assigned"], resource_type="vehicle", source="vehicle_schedule_assigned")
    driver_schedule_assigned_long = preprocess_availability_wide(raw_tables["driver_schedule_assigned"], resource_type="driver", source="driver_schedule_assigned")
    expert_schedule = preprocess_expert_schedule(raw_tables["expert_best_schedule"])

    driver_availability = _attach_driver_ids(driver_availability, drivers)
    driver_schedule_assigned_long = _attach_driver_ids(driver_schedule_assigned_long, drivers)

    tables = {
        "requests": requests,
        "vehicles": vehicles,
        "drivers": drivers,
        "vehicle_availability": vehicle_availability,
        "driver_availability": driver_availability,
        "vehicle_schedule_assigned_long": vehicle_schedule_assigned_long,
        "driver_schedule_assigned_long": driver_schedule_assigned_long,
        "expert_schedule": expert_schedule,
    }
    return tables


def write_processed_tables(tables: Dict[str, pd.DataFrame], processed_dir: Path = PROCESSED_DIR, validation_dir: Path = VALIDATION_REPORT_DIR) -> None:
    processed_dir = Path(processed_dir)
    validation_dir = Path(validation_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        if key == "validation_report":
            continue
        filename = STANDARD_TABLE_FILES.get(key, f"{key}.csv")
        df.to_csv(processed_dir / filename, index=False, encoding="utf-8-sig")
    if "validation_report" in tables:
        tables["validation_report"].to_csv(validation_dir / STANDARD_TABLE_FILES["validation_report"], index=False, encoding="utf-8-sig")
        # Excel report is more convenient for non-programmer inspection.
        tables["validation_report"].to_excel(validation_dir / "validation_report.xlsx", index=False)


def run_preprocessing(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR, validation_dir: Path = VALIDATION_REPORT_DIR) -> Dict[str, pd.DataFrame]:
    raw_tables = load_raw_inputs(Path(raw_dir))
    tables = build_processed_tables(raw_tables)
    tables["validation_report"] = validate_tables(tables)
    write_processed_tables(tables, Path(processed_dir), Path(validation_dir))
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G-LIFT preprocessing pipeline")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory containing raw Excel files")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="Directory to write processed CSV files")
    parser.add_argument("--validation-dir", default=str(VALIDATION_REPORT_DIR), help="Directory to write validation reports")
    args = parser.parse_args()
    tables = run_preprocessing(Path(args.raw_dir), Path(args.processed_dir), Path(args.validation_dir))
    print("G-LIFT preprocessing completed.")
    for name, df in tables.items():
        print(f"- {name}: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
