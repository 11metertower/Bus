"""Unit tests for the independent hard-constraint checker."""

import unittest

import pandas as pd

from src.hard_constraint_checker import (
    _overlap,
    hard_violation_total,
    run_hard_constraint_checks,
)


def _violation_count(report: pd.DataFrame, issue_type: str) -> int:
    row = report[report["issue_type"] == issue_type]
    return int(row["violation_count"].iloc[0]) if not row.empty else -1


class TestOverlap(unittest.TestCase):
    def test_adjacent_intervals_do_not_overlap(self):
        self.assertFalse(_overlap(0, 5, 5, 10))  # end-exclusive

    def test_true_overlap(self):
        self.assertTrue(_overlap(0, 6, 5, 10))

    def test_disjoint(self):
        self.assertFalse(_overlap(0, 5, 6, 10))


class TestRunHardConstraintChecks(unittest.TestCase):
    def _tables(self):
        return {
            "vehicles": pd.DataFrame(
                {"vehicle_id": ["2202"], "vehicle_class": ["승용차"], "is_maintenance_or_unavailable": [False]}
            ),
            "drivers": pd.DataFrame({"driver_id": ["DRV_001"]}),
            "requests": pd.DataFrame({"request_id": ["R1", "R2"], "request_type": ["영내", "영내"]}),
            "vehicle_availability": pd.DataFrame(),
            "driver_availability": pd.DataFrame(),
        }

    def test_empty_schedule_flags_empty(self):
        report = run_hard_constraint_checks(pd.DataFrame(), self._tables())
        self.assertEqual(_violation_count(report, "EMPTY_SCHEDULE"), 1)

    def test_vehicle_time_overlap_detected(self):
        schedule = pd.DataFrame(
            {
                "request_id": ["R1", "R2"],
                "status": ["ASSIGNED", "ASSIGNED"],
                "assignment_mode": ["OTHER_DIRECT_DRIVER", "OTHER_DIRECT_DRIVER"],
                "assigned_vehicle_id": ["2202", "2202"],
                "assigned_driver_id": ["", ""],
                "required_vehicle_class": ["승용차", "승용차"],
                "request_type": ["영내", "영내"],
                "start_min": [0, 30],
                "end_min": [60, 90],
            }
        )
        report = run_hard_constraint_checks(schedule, self._tables())
        self.assertEqual(_violation_count(report, "VEHICLE_TIME_OVERLAP"), 1)

    def test_non_overlapping_same_vehicle_is_clean(self):
        schedule = pd.DataFrame(
            {
                "request_id": ["R1", "R2"],
                "status": ["ASSIGNED", "ASSIGNED"],
                "assignment_mode": ["OTHER_DIRECT_DRIVER", "OTHER_DIRECT_DRIVER"],
                "assigned_vehicle_id": ["2202", "2202"],
                "assigned_driver_id": ["", ""],
                "required_vehicle_class": ["승용차", "승용차"],
                "request_type": ["영내", "영내"],
                "start_min": [0, 60],
                "end_min": [60, 120],
            }
        )
        report = run_hard_constraint_checks(schedule, self._tables())
        self.assertEqual(_violation_count(report, "VEHICLE_TIME_OVERLAP"), 0)

    def test_other_driver_misassignment_detected(self):
        schedule = pd.DataFrame(
            {
                "request_id": ["R1"],
                "status": ["ASSIGNED"],
                "assignment_mode": ["OTHER_DIRECT_DRIVER"],
                "assigned_vehicle_id": ["2202"],
                "assigned_driver_id": ["DRV_001"],  # Other must not carry a driver
                "required_vehicle_class": ["승용차"],
                "request_type": ["영내"],
                "start_min": [0],
                "end_min": [60],
            }
        )
        report = run_hard_constraint_checks(schedule, self._tables())
        self.assertEqual(_violation_count(report, "OTHER_DRIVER_MISASSIGNED"), 1)

    def test_duplicate_request_row_detected(self):
        schedule = pd.DataFrame(
            {
                "request_id": ["R1", "R1"],
                "status": ["ASSIGNED", "ASSIGNED"],
                "assignment_mode": ["OTHER_DIRECT_DRIVER", "OTHER_DIRECT_DRIVER"],
                "assigned_vehicle_id": ["2202", "2202"],
                "assigned_driver_id": ["", ""],
                "required_vehicle_class": ["승용차", "승용차"],
                "request_type": ["영내", "영내"],
                "start_min": [0, 200],
                "end_min": [60, 260],
            }
        )
        report = run_hard_constraint_checks(schedule, self._tables())
        self.assertEqual(_violation_count(report, "REQUEST_ROW_COUNT_NOT_ONE"), 1)


class TestHardViolationTotal(unittest.TestCase):
    def test_sums_only_error_rows(self):
        report = pd.DataFrame(
            [
                {"severity": "ERROR", "violation_count": 2},
                {"severity": "OK", "violation_count": 0},
                {"severity": "ERROR", "violation_count": 3},
            ]
        )
        self.assertEqual(hard_violation_total(report), 5)

    def test_empty_report_is_zero(self):
        self.assertEqual(hard_violation_total(pd.DataFrame()), 0)


if __name__ == "__main__":
    unittest.main()
