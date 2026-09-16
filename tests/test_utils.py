"""Unit tests for src.utils parsing/normalization helpers."""

import unittest
from datetime import time, timedelta

from src.utils import (
    availability_to_bool,
    clean_text,
    default_priority_from_request_type,
    is_missing,
    make_driver_id,
    make_request_id,
    minutes_to_hhmm,
    normalize_transmission,
    normalize_vehicle_class,
    normalize_vehicle_number,
    parse_skill_code,
    parse_time_to_minutes,
)


class TestIsMissing(unittest.TestCase):
    def test_none_and_nan_are_missing(self):
        self.assertTrue(is_missing(None))
        self.assertTrue(is_missing(float("nan")))

    def test_missing_like_strings(self):
        for value in ["", "  ", "nan", "NA", "n/a", "-", "--", "None", "null"]:
            self.assertTrue(is_missing(value), value)

    def test_real_values_are_present(self):
        self.assertFalse(is_missing("2202"))
        self.assertFalse(is_missing(0))
        self.assertFalse(is_missing("장거리"))


class TestCleanText(unittest.TestCase):
    def test_collapses_whitespace_and_ideographic_space(self):
        self.assertEqual(clean_text("  a　 b  "), "a b")

    def test_missing_returns_none(self):
        self.assertIsNone(clean_text("na"))


class TestNormalizeVehicleNumber(unittest.TestCase):
    def test_strips_trailing_dot_zero(self):
        self.assertEqual(normalize_vehicle_number("2202.0"), "2202")

    def test_plain_number_unchanged(self):
        self.assertEqual(normalize_vehicle_number("2202"), "2202")

    def test_removes_inner_spaces(self):
        self.assertEqual(normalize_vehicle_number(" 22 02 "), "2202")

    def test_missing_returns_none(self):
        self.assertIsNone(normalize_vehicle_number("-"))


class TestNormalizeVehicleClass(unittest.TestCase):
    def test_known_synonyms_map_to_canonical(self):
        self.assertEqual(normalize_vehicle_class("1톤트럭"), "1톤 원캡 트럭")
        self.assertEqual(normalize_vehicle_class("스타리아"), "소형 버스(스타리아)")
        self.assertEqual(normalize_vehicle_class("승용차"), "승용차")

    def test_unknown_class_kept_as_is(self):
        self.assertEqual(normalize_vehicle_class("특수차량"), "특수차량")


class TestNormalizeTransmission(unittest.TestCase):
    def test_maps_auto_and_manual(self):
        self.assertEqual(normalize_transmission("자동변속"), "자동")
        self.assertEqual(normalize_transmission("수동"), "수동")

    def test_missing_returns_none(self):
        self.assertIsNone(normalize_transmission(""))


class TestParseTimeToMinutes(unittest.TestCase):
    def test_string_hhmm(self):
        self.assertEqual(parse_time_to_minutes("08:00"), 480)
        self.assertEqual(parse_time_to_minutes("08:00:00"), 480)

    def test_midnight_end_of_day(self):
        self.assertEqual(parse_time_to_minutes("24:00"), 1440)

    def test_datetime_time_object(self):
        self.assertEqual(parse_time_to_minutes(time(8, 30)), 510)

    def test_excel_day_fraction(self):
        self.assertEqual(parse_time_to_minutes(0.5), 720)

    def test_timedelta(self):
        self.assertEqual(parse_time_to_minutes(timedelta(hours=1, minutes=15)), 75)

    def test_embedded_time_token(self):
        self.assertEqual(parse_time_to_minutes("OFF (08:00 - 24:00)"), 480)

    def test_missing_returns_none(self):
        self.assertIsNone(parse_time_to_minutes(None))


class TestMinutesToHhmm(unittest.TestCase):
    def test_roundtrip_values(self):
        self.assertEqual(minutes_to_hhmm(480), "08:00")
        self.assertEqual(minutes_to_hhmm(1440), "24:00")
        self.assertEqual(minutes_to_hhmm(75), "01:15")

    def test_none_returns_none(self):
        self.assertIsNone(minutes_to_hhmm(None))


class TestParseSkillCode(unittest.TestCase):
    def test_valid_three_digit_code(self):
        result = parse_skill_code("233")
        self.assertEqual(result["skill_code"], "233")
        self.assertEqual((result["skill_small"], result["skill_medium"], result["skill_large"]), (2, 3, 3))
        self.assertTrue(result["skill_valid"])

    def test_parenthesized_code(self):
        self.assertEqual(parse_skill_code("(111)")["skill_code"], "111")

    def test_short_code_is_left_padded_and_flagged(self):
        result = parse_skill_code("12")
        self.assertEqual(result["skill_code"], "012")
        self.assertFalse(result["skill_valid"])  # 0 is out of the 1..4 range
        self.assertEqual(result["skill_warning"], "skill_code_left_padded_to_3_digits")

    def test_missing_code(self):
        result = parse_skill_code(None)
        self.assertFalse(result["skill_valid"])
        self.assertEqual(result["skill_warning"], "missing_skill")


class TestAvailabilityToBool(unittest.TestCase):
    def test_t_and_f(self):
        self.assertTrue(availability_to_bool("T"))
        self.assertFalse(availability_to_bool("f"))

    def test_unknown_returns_none(self):
        self.assertIsNone(availability_to_bool("x"))
        self.assertIsNone(availability_to_bool(""))


class TestDefaultPriority(unittest.TestCase):
    def test_priority_by_keyword(self):
        self.assertEqual(default_priority_from_request_type("장거리 지원"), 3)
        self.assertEqual(default_priority_from_request_type("단거리"), 2)
        self.assertEqual(default_priority_from_request_type("영내 이동"), 1)

    def test_unknown_defaults_to_one(self):
        self.assertEqual(default_priority_from_request_type("기타"), 1)
        self.assertEqual(default_priority_from_request_type(None), 1)


class TestIdMakers(unittest.TestCase):
    def test_zero_padded_ids(self):
        self.assertEqual(make_driver_id(3), "DRV_003")
        self.assertEqual(make_request_id(42), "REQ_042")


if __name__ == "__main__":
    unittest.main()
