"""Unit tests for scoring/feature helpers in src.feature_extractor."""

import unittest

import pandas as pd

from src.feature_extractor import (
    as_key,
    build_availability_index,
    driver_has_required_skill,
    infer_required_scope_level,
    infer_skill_slot,
    is_resource_available,
    make_expert_maps,
    request_slots,
    summarize_candidate,
    to_bool,
    vehicle_fit_score,
    vehicle_matches_request,
)


class TestRequestSlots(unittest.TestCase):
    def test_end_is_exclusive(self):
        # [0, 5) occupies slot 0 only; [5, 10) occupies slot 1 -> no overlap.
        self.assertEqual(request_slots(0, 5), [0])
        self.assertEqual(request_slots(5, 10), [1])

    def test_multi_slot_span(self):
        self.assertEqual(request_slots(0, 10), [0, 1])

    def test_zero_or_negative_duration_is_empty(self):
        self.assertEqual(request_slots(10, 10), [])
        self.assertEqual(request_slots(10, 5), [])

    def test_missing_is_empty(self):
        self.assertEqual(request_slots(None, 5), [])


class TestScopeAndSkillSlot(unittest.TestCase):
    def test_required_scope_level(self):
        self.assertEqual(infer_required_scope_level("장거리"), 1)
        self.assertEqual(infer_required_scope_level("단거리"), 2)
        self.assertEqual(infer_required_scope_level("영내"), 3)

    def test_skill_slot_from_skill_text(self):
        self.assertEqual(infer_skill_slot("대형", None), "skill_large")
        self.assertEqual(infer_skill_slot("중형", None), "skill_medium")
        self.assertEqual(infer_skill_slot("소형", None), "skill_small")

    def test_skill_slot_falls_back_to_vehicle_class(self):
        self.assertEqual(infer_skill_slot("", "9톤 화물차"), "skill_large")
        self.assertEqual(infer_skill_slot("", "1톤 원캡 트럭"), "skill_medium")
        self.assertEqual(infer_skill_slot("", "승용차"), "skill_small")


class TestDriverHasRequiredSkill(unittest.TestCase):
    def _driver(self, **kw):
        base = {"skill_valid": True, "skill_small": 1, "skill_medium": 1, "skill_large": 1}
        base.update(kw)
        return pd.Series(base)

    def test_capable_driver_passes(self):
        driver = self._driver(skill_large=1)
        vehicle = pd.Series({"required_skill_text": "대형", "vehicle_class": "9톤 화물차"})
        request = pd.Series({"request_type": "장거리"})  # required level 1
        self.assertTrue(driver_has_required_skill(driver, vehicle, request))

    def test_insufficient_skill_fails(self):
        driver = self._driver(skill_large=4)  # 4 = cannot drive
        vehicle = pd.Series({"required_skill_text": "대형", "vehicle_class": "9톤 화물차"})
        request = pd.Series({"request_type": "영내"})  # required level 3
        self.assertFalse(driver_has_required_skill(driver, vehicle, request))

    def test_invalid_skill_flag_fails(self):
        driver = self._driver(skill_valid=False)
        vehicle = pd.Series({"required_skill_text": "소형", "vehicle_class": "승용차"})
        request = pd.Series({"request_type": "영내"})
        self.assertFalse(driver_has_required_skill(driver, vehicle, request))


class TestVehicleMatchesRequest(unittest.TestCase):
    def test_exact_class_match(self):
        vehicle = pd.Series({"vehicle_class": "승용차"})
        request = pd.Series({"required_vehicle_class": "승용차"})
        self.assertTrue(vehicle_matches_request(vehicle, request))

    def test_mismatch(self):
        vehicle = pd.Series({"vehicle_class": "지프"})
        request = pd.Series({"required_vehicle_class": "승용차"})
        self.assertFalse(vehicle_matches_request(vehicle, request))


class TestAvailabilityIndex(unittest.TestCase):
    def test_index_only_keeps_available_slots(self):
        avail = pd.DataFrame(
            {
                "resource_id": ["2202", "2202", "2202"],
                "slot_index": [0, 1, 2],
                "is_available": [True, False, True],
            }
        )
        index = build_availability_index(avail, "resource_id")
        self.assertEqual(index["2202"], {0, 2})

    def test_is_resource_available_requires_all_slots(self):
        index = {"2202": {0, 1, 2}}
        self.assertTrue(is_resource_available("2202", [0, 1], index))
        self.assertFalse(is_resource_available("2202", [1, 5], index))

    def test_empty_slots_is_unavailable(self):
        self.assertFalse(is_resource_available("2202", [], {"2202": {0}}))


class TestVehicleFitScore(unittest.TestCase):
    def test_base_score_for_neutral_request(self):
        vehicle = pd.Series({"vehicle_id": "2202", "vehicle_class": "승용차"})
        request = pd.Series({"request_type": "영내", "requested_vehicle_id": None})
        self.assertEqual(vehicle_fit_score(vehicle, request), 70)

    def test_requested_vehicle_bonus(self):
        vehicle = pd.Series({"vehicle_id": "2202", "vehicle_class": "승용차"})
        request = pd.Series({"request_type": "영내", "requested_vehicle_id": "2202"})
        self.assertEqual(vehicle_fit_score(vehicle, request), 85)

    def test_score_is_clamped_0_100(self):
        vehicle = pd.Series({"vehicle_id": "2202", "vehicle_class": "고속버스"})
        request = pd.Series({"request_type": "장거리", "requested_vehicle_id": "2202"})
        self.assertLessEqual(vehicle_fit_score(vehicle, request), 100)


class TestHelpers(unittest.TestCase):
    def test_to_bool_variants(self):
        self.assertTrue(to_bool("true"))
        self.assertTrue(to_bool("1"))
        self.assertTrue(to_bool(True))
        self.assertFalse(to_bool("no"))
        self.assertFalse(to_bool(None))

    def test_as_key_normalizes_numeric_ids(self):
        self.assertEqual(as_key(2202), "2202")
        self.assertEqual(as_key("2202.0"), "2202")
        self.assertIsNone(as_key(None))

    def test_summarize_candidate_counts(self):
        schedule = pd.DataFrame(
            {
                "status": ["ASSIGNED", "ASSIGNED", "UNSERVED"],
                "assignment_mode": ["MILITARY_DRIVER_REQUIRED", "OTHER_DIRECT_DRIVER", ""],
            }
        )
        summary = summarize_candidate(schedule)
        self.assertEqual(summary["assigned_count"], 2)
        self.assertEqual(summary["unserved_count"], 1)
        self.assertEqual(summary["other_assigned_count"], 1)

    def test_make_expert_maps(self):
        expert = pd.DataFrame(
            {
                "request_id": ["REQ_001", "REQ_002"],
                "assigned_vehicle_id": ["2202", "3303"],
                "assigned_driver_raw": ["김병장", "Other"],
            }
        )
        vehicle_map, driver_map = make_expert_maps(expert)
        self.assertEqual(vehicle_map["REQ_001"], "2202")
        self.assertEqual(driver_map["REQ_002"], "Other")


if __name__ == "__main__":
    unittest.main()
