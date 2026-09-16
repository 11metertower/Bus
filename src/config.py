"""Project-wide paths and constants for G-LIFT MVP.

The project intentionally keeps raw Excel files, processed CSV files, and
human-readable reports separate so that the preprocessing step is auditable.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
HISTORICAL_DIR = DATA_DIR / "historical"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"
VALIDATION_REPORT_DIR = REPORTS_DIR / "validation"

TIME_SLOT_MINUTES = 5

RAW_FILE_NAMES = {
    "requests_raw": "requests_raw.xlsx",
    "vehicles": "vehicles.xlsx",
    "vehicle_schedule": "vehicle_schedule.xlsx",
    "vehicle_schedule_assigned": "vehicle_schedule_assigned.xlsx",
    "drivers": "drivers.xlsx",
    "driver_availability": "driver_availability.xlsx",
    "driver_schedule_assigned": "driver_schedule_assigned.xlsx",
    "expert_best_schedule": "expert_best_schedule.xlsx",
}

STANDARD_TABLE_FILES = {
    "requests": "requests.csv",
    "vehicles": "vehicles.csv",
    "drivers": "drivers.csv",
    "vehicle_availability": "vehicle_availability.csv",
    "driver_availability": "driver_availability.csv",
    "vehicle_schedule_assigned_long": "vehicle_schedule_assigned_long.csv",
    "driver_schedule_assigned_long": "driver_schedule_assigned_long.csv",
    "expert_schedule": "expert_schedule.csv",
    "validation_report": "validation_report.csv",
}

# Scenario and optimizer-stabilization paths
SCENARIOS_DIR = DATA_DIR / "scenarios"
EVALUATION_REPORT_DIR = REPORTS_DIR / "evaluation"

# Future regular-route grouping hooks.  The current sample data does not
# provide stable request IDs for these groups, so they are empty by default.
HARD_SAME_DRIVER_GROUPS = []
SOFT_SAME_DRIVER_GROUPS = []
SOFT_SHUTTLE_GROUP_RULES = []

SCENARIO_NAMES = [
    "baseline",
    "vehicle_shortage",
    "driver_shortage",
    "skill_shortage",
    "other_vehicle_conflict",
    "mixed_shortage",
]

# Single source of truth for the expert-designed baseline weights over the 12
# candidate-quality features.  Shared by the rule-based Ranker
# (src.ranker.apply_rule_based_ranker) and by the inverse-optimization prior
# (src.inverse_optimization.learn_objective_weights).  Keep these summing to 1.
BASELINE_FEATURE_WEIGHTS = {
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
