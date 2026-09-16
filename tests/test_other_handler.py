"""Unit tests for the 'Other' direct-driver handling rules."""

import unittest

import pandas as pd

from src.other_handler import (
    MILITARY_DRIVER_MODE,
    OTHER_DRIVER_MODE,
    add_driver_mode_columns,
    is_other_driver_value,
)


class TestIsOtherDriverValue(unittest.TestCase):
    def test_other_is_case_insensitive(self):
        self.assertTrue(is_other_driver_value("Other"))
        self.assertTrue(is_other_driver_value("other"))
        self.assertTrue(is_other_driver_value("  OTHER  "))

    def test_military_driver_name_is_not_other(self):
        self.assertFalse(is_other_driver_value("홍길동"))

    def test_missing_is_not_other(self):
        self.assertFalse(is_other_driver_value(None))
        self.assertFalse(is_other_driver_value(""))


class TestAddDriverModeColumns(unittest.TestCase):
    def test_marks_other_and_military_rows(self):
        df = pd.DataFrame({"driver_raw": ["Other", "김병장", "other"]})
        out = add_driver_mode_columns(df)
        self.assertEqual(out["is_other"].tolist(), [True, False, True])
        self.assertEqual(out["requires_driver"].tolist(), [False, True, False])
        self.assertEqual(
            out["driver_mode"].tolist(),
            [OTHER_DRIVER_MODE, MILITARY_DRIVER_MODE, OTHER_DRIVER_MODE],
        )

    def test_missing_driver_column_defaults_to_military(self):
        df = pd.DataFrame({"request_id": ["REQ_001"]})
        out = add_driver_mode_columns(df)
        self.assertFalse(bool(out["is_other"].iloc[0]))
        self.assertTrue(bool(out["requires_driver"].iloc[0]))
        self.assertEqual(out["driver_mode"].iloc[0], MILITARY_DRIVER_MODE)


if __name__ == "__main__":
    unittest.main()
