# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data-only regressions for the common Indic ITN prompt/example contract.

These checks need no inference runtime and can run with pytest --noconftest.
They verify the approved formatting contract, not native-speaker accuracy.
"""

import json
import re
import unicodedata
from decimal import Decimal
from pathlib import Path

import pytest

PROMPT_DIR = Path(__file__).parents[4] / "nemo_curator" / "stages" / "audio" / "text_filtering" / "prompts"
EXAMPLES = json.loads((PROMPT_DIR / "itn_language_examples.json").read_text(encoding="utf-8"))
PROMPT = (PROMPT_DIR / "itn_prompt_indic.md").read_text(encoding="utf-8")
LANGUAGES = (
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
    "ur",
    "brx",
    "doi",
    "kok",
    "ks",
    "mai",
    "mni",
    "ne",
    "sa",
    "sat",
    "sd",
)
CATEGORIES = (
    "Cardinal",
    "Ordinal",
    "Date",
    "Time",
    "Duration",
    "Money",
    "Percent",
    "Units",
    "Fractions",
    "Phone",
    "URL/Email",
    "Titles",
    "Address",
    "Roman num.",
    "Negative",
    "Decades",
    "Letter+num",
)


def _rows(language: str) -> dict[str, tuple[str, str]]:
    result = {}
    for line in EXAMPLES[language].splitlines():
        if not line.startswith("|") or line.startswith(("| Category", "|---")):
            continue
        category, spoken, written = (cell.strip() for cell in line.strip("|").split("|"))
        assert category not in result
        result[category] = (spoken, written)
    return result


def test_exact_language_coverage() -> None:
    assert tuple(EXAMPLES) == LANGUAGES


@pytest.mark.parametrize("language", LANGUAGES)
def test_category_and_pair_coverage(language: str) -> None:
    rows = _rows(language)
    assert tuple(rows) == CATEGORIES
    for category, (spoken, written) in rows.items():
        assert len(spoken.split(" / ")) == len(written.split(" / ")), category


@pytest.mark.parametrize("language", LANGUAGES)
def test_newly_converted_numbers_use_ascii_digits(language: str) -> None:
    for _, written in _rows(language).values():
        assert all(not c.isdecimal() or c in "0123456789" for c in written)


@pytest.mark.parametrize("language", LANGUAGES)
def test_cardinal_grouping_preserves_values(language: str) -> None:
    written = _rows(language)["Cardinal"][1]
    assert written == "14 / 1,030.5 / 2,024"
    values = [Decimal(value.replace(",", "")) for value in written.split(" / ")]
    assert values == [Decimal(14), Decimal("1030.5"), Decimal(2024)]
    assert "1,00,000" in PROMPT
    assert "1,23,456.78" in PROMPT
    assert "Do not group years" in PROMPT


@pytest.mark.parametrize("language", LANGUAGES)
def test_no_english_ordinal_or_decade_suffix(language: str) -> None:
    rows = _rows(language)
    for category, (_, written) in rows.items():
        assert not re.search(r"\d(?:st|nd|rd|th|s)\b", written), (language, category)
    for value in rows["Ordinal"][1].split(" / "):
        assert re.fullmatch(r"\d+[^\d]+", value), value
        assert any(unicodedata.category(c).startswith("L") and not c.isascii() for c in value), value
    assert any(not c.isascii() for c in rows["Decades"][1])


@pytest.mark.parametrize("language", LANGUAGES)
def test_dates_keep_native_components_without_year_comma(language: str) -> None:
    spoken, written = _rows(language)["Date"]
    assert "," not in written
    assert "،" not in written
    for index, value in enumerate(written.split(" / ")):
        assert re.search(r"(?<!\d)22(?!\d)", value)
        assert value.endswith(("1847", "1990")[index])
        assert not re.search(r"\b0\d\b", value)
    # Existing month-first languages must not be forced into day-first order.
    if language in {"kn", "ml", "or", "ta", "te", "brx", "kok", "ks", "mni", "sat", "sd", "sa"}:
        assert spoken.split()[0] == written.split()[0]


@pytest.mark.parametrize("language", LANGUAGES)
def test_clocks_and_durations_share_the_common_numeric_style(language: str) -> None:
    rows = _rows(language)
    clocks = rows["Time"][1].split(" / ")
    assert clocks[:3] == ["3:05 PM", "10:00 AM", "1:45"]
    assert any(not c.isascii() for c in clocks[3])
    assert rows["Duration"][1] == "1:00"
    assert "9:00:02" in PROMPT
    assert "more than 23 hours" in PROMPT
    assert "only when seconds are supplied" in PROMPT
    assert "do not infer it" in PROMPT
    assert "do not globally reformat" in PROMPT


@pytest.mark.parametrize("language", LANGUAGES)
def test_money_and_percent_symbols_have_no_number_gap(language: str) -> None:
    rows = _rows(language)
    assert rows["Money"][1] == "$52 / $249.99"
    for category in ("Money", "Percent"):
        value = rows[category][1]
        assert not re.search(r"[$₹£€]\s+\d|\d\s+%", value)
    assert "₹625.02" in PROMPT
    assert "carry excess minor units" in PROMPT


@pytest.mark.parametrize("language", LANGUAGES)
def test_phone_examples_are_contiguous_and_not_grouped(language: str) -> None:
    phones = _rows(language)["Phone"][1].split(" / ")
    assert phones == ["5558675309", "18005550199"]
    assert all(re.fullmatch(r"\d+", value) for value in phones)
    assert "preserving leading zeroes" in PROMPT
    assert "+<country code> <national number>" in PROMPT
    assert "Do not invent a country code" in PROMPT


def test_native_case_and_suffix_examples() -> None:
    expected = {
        "kok": "1लो / 21वो / 50वो",
        "or": "1ମ / 21ତମ / 50ତମ",
        "sd": "1यों / 21हों / 50हों",
        "ml": "1-ാമത്തെ / 21-ാമത്തെ / 50-ാമത്തെ",
        "sa": "1तमम् / 21तमम् / 50तमम्",
        "mni": "1ꯁꯨꯕ / 21ꯁꯨꯕ / 50ꯁꯨꯕ",
    }
    for language, written in expected.items():
        assert _rows(language)["Ordinal"][1] == written
    assert "22वी" in _rows("kok")["Date"][1]
    assert "22ତମ" in _rows("or")["Date"][1]
    assert "22तमे दिने" in _rows("sa")["Date"][1]
    assert "इक्कीमां / पंजाहमां" in _rows("doi")["Ordinal"][0]


def test_corrected_examples_preserve_surrounding_words() -> None:
    assert _rows("ml")["Roman num."][1].startswith("ഹെൻറി VIII രാജാവ് / ")
    assert _rows("ta")["Percent"][1] == "0.5% / 20% முதல் 30%"
    assert "ఒకటిన్నర గంటలు → `1:30`" in EXAMPLES["te"]
    assert "ఇద్దరు పిల్లలకి → `ఇద్దరు పిల్లలకి`" in EXAMPLES["te"]
    assert "అతను డాక్టర్ → `అతను డాక్టర్`" in EXAMPLES["te"]


def test_santali_and_sindhi_years_have_explicit_scale_words() -> None:
    assert "ᱢᱤᱫ ᱜᱮᱥᱟᱭ ᱤᱨᱟᱹᱞ ᱥᱟᱭ" in _rows("sat")["Date"][0]
    assert "ᱢᱤᱫ ᱜᱮᱥᱟᱭ ᱟᱨᱮ ᱥᱟᱭ" in _rows("sat")["Date"][0]
    assert "अरिड़हं सौ सतेतालीह" in _rows("sd")["Date"][0]
    assert "उणीह सौ नवे" in _rows("sd")["Date"][0]
