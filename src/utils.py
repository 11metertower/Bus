"""Utility functions for G-LIFT preprocessing.

The utilities are intentionally defensive: the current MVP data is clean and
virtual, but later versions may include partially typed Excel values, hidden
spaces, numeric vehicle IDs, or Excel time fractions.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, time, timedelta
from typing import Any, Optional, Tuple

import pandas as pd

MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "-", "--"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except TypeError:
        pass
    text = str(value).strip()
    return text.lower() in MISSING_STRINGS


def clean_text(value: Any) -> Optional[str]:
    """Return stripped text, or None for missing-like values."""
    if is_missing(value):
        return None
    text = str(value).replace("\u3000", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text or None


def normalize_vehicle_number(value: Any) -> Optional[str]:
    """Normalize vehicle number to a string without trailing .0.

    Current sample data uses short virtual numbers such as 2202. Future data may
    contain real-style strings, so the function does not force numeric-only.
    """
    text = clean_text(value)
    if text is None:
        return None
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    # Remove spaces inside virtual numeric vehicle IDs, keep Korean letters if any.
    text = re.sub(r"\s+", "", text)
    return text


def normalize_vehicle_class(value: Any) -> Optional[str]:
    """Normalize vehicle class names while preserving recognizable Korean labels."""
    text = clean_text(value)
    if text is None:
        return None
    compact = re.sub(r"[\s_\-]+", "", text)
    compact = compact.replace("(", "").replace(")", "")
    mapping = {
        "승용차": "승용차",
        "지프": "지프",
        "1톤원캡트럭": "1톤 원캡 트럭",
        "1톤트럭": "1톤 원캡 트럭",
        "5톤화물차": "5톤 화물차",
        "5톤급유차": "5톤 급유차",
        "9톤화물차": "9톤 화물차",
        "소형버스스타리아": "소형 버스(스타리아)",
        "소형버스": "소형 버스(스타리아)",
        "스타리아": "소형 버스(스타리아)",
        "고속버스": "고속버스",
        "버스": "버스",
    }
    return mapping.get(compact, text)


def normalize_transmission(value: Any) -> Optional[str]:
    text = clean_text(value)
    if text is None:
        return None
    if "자동" in text:
        return "자동"
    if "수동" in text:
        return "수동"
    return text


def parse_time_to_minutes(value: Any) -> Optional[int]:
    """Parse Excel/CSV time values to minutes from 00:00.

    Supports datetime.time, pandas Timestamp, Excel day fractions, and strings
    like '08:00', '08:00:00', '24:00'.
    """
    if is_missing(value):
        return None
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    if isinstance(value, pd.Timestamp):
        return value.hour * 60 + value.minute
    if isinstance(value, timedelta):
        return int(value.total_seconds() // 60)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Excel stores time-only values as a fraction of a day.
        if 0 <= float(value) <= 1:
            minutes = int(round(float(value) * 24 * 60))
            return min(minutes, 24 * 60)
        # If a user already supplied minutes, accept it.
        if 0 <= float(value) <= 24 * 60:
            return int(round(float(value)))
    text = clean_text(value)
    if text is None:
        return None
    # Extract the first time-like token from text such as OFF (00:00 - 24:00).
    m = re.search(r"(\d{1,2})\s*:\s*(\d{2})(?:\s*:\s*\d{2})?", text)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2))
    if hour == 24 and minute == 0:
        return 24 * 60
    if 0 <= hour < 24 and 0 <= minute < 60:
        return hour * 60 + minute
    return None


def minutes_to_hhmm(minutes: Optional[int]) -> Optional[str]:
    if minutes is None:
        return None
    minutes = int(minutes)
    if minutes == 24 * 60:
        return "24:00"
    minutes = minutes % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def parse_skill_code(value: Any) -> dict:
    """Parse driver skill code such as 111, 233, 333.

    Excel may read '(111)' or '111' as an integer. If a value has fewer than
    three digits, it is left-padded and flagged as invalid when 0 appears.
    """
    raw = clean_text(value)
    result = {
        "skill_raw": raw,
        "skill_code": None,
        "skill_small": None,
        "skill_medium": None,
        "skill_large": None,
        "skill_valid": False,
        "skill_warning": None,
    }
    if raw is None:
        result["skill_warning"] = "missing_skill"
        return result
    digits = "".join(re.findall(r"\d", raw))
    if not digits:
        result["skill_warning"] = "no_digits_in_skill"
        return result
    if len(digits) < 3:
        digits = digits.zfill(3)
        result["skill_warning"] = "skill_code_left_padded_to_3_digits"
    elif len(digits) > 3:
        digits = digits[-3:]
        result["skill_warning"] = "skill_code_truncated_to_last_3_digits"
    values = [int(ch) for ch in digits]
    result.update({
        "skill_code": digits,
        "skill_small": values[0],
        "skill_medium": values[1],
        "skill_large": values[2],
        "skill_valid": all(1 <= v <= 4 for v in values),
    })
    if not result["skill_valid"] and result["skill_warning"] is None:
        result["skill_warning"] = "skill_level_outside_1_to_4"
    return result


def make_driver_id(index: int) -> str:
    return f"DRV_{index:03d}"


def make_request_id(index: int) -> str:
    return f"REQ_{index:03d}"


def availability_to_bool(value: Any) -> Optional[bool]:
    text = clean_text(value)
    if text is None:
        return None
    upper = text.upper()
    if upper == "T":
        return True
    if upper == "F":
        return False
    return None


def default_priority_from_request_type(request_type: Any) -> int:
    text = clean_text(request_type) or ""
    if "장거리" in text:
        return 3
    if "단거리" in text:
        return 2
    if "영내" in text:
        return 1
    return 1
