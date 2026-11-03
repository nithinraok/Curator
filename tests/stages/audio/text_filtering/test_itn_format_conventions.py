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
    expected_categories = CATEGORIES
    if language == "mr":
        expected_categories = (*CATEGORIES[:1], "Decimal digit sequence", *CATEGORIES[1:])
    assert tuple(rows) == expected_categories
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
def test_clocks_keep_colon_notation(language: str) -> None:
    rows = _rows(language)
    clocks = rows["Time"][1].split(" / ")
    assert clocks[:3] == ["3:05 PM", "10:00 AM", "1:45"]
    assert any(not c.isascii() for c in clocks[3])
    assert "only when seconds are supplied" in PROMPT
    assert "do not infer it" in PROMPT
    assert "do not globally reformat" in PROMPT


@pytest.mark.parametrize("language", LANGUAGES)
def test_elapsed_durations_preserve_native_units_and_order(language: str) -> None:
    spoken, written = _rows(language)["Duration"]
    if language == "hi":
        expected = "1 घंटा / 1 घंटा 5 सेकंड / 1 मिनट 5 सेकंड"
    else:
        words = spoken.split(" ")
        words[-1 if language == "mni" else 0] = "1"
        expected = " ".join(words)
    assert written == expected
    assert ":" not in written
    assert "retain the native duration-unit words" in PROMPT
    assert "Do not render an elapsed duration as `H:MM` or `H:MM:SS`" in PROMPT
    assert "Genuine clock readings continue to follow the clock rules above" in PROMPT
    assert "currency or duration rule" not in PROMPT


@pytest.mark.parametrize("language", LANGUAGES)
def test_money_and_percent_symbols_have_no_number_gap(language: str) -> None:
    rows = _rows(language)
    expected_money = "$52 / $249.99 / ₹52.04" if language == "hi" else "$52 / $249.99"
    assert rows["Money"][1] == expected_money
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
    assert "ఒకటిన్నర గంటలు → `1.5 గంటలు`" in EXAMPLES["te"]
    assert "ఇద్దరు పిల్లలకి → `ఇద్దరు పిల్లలకి`" in EXAMPLES["te"]
    assert "అతను డాక్టర్ → `అతను డాక్టర్`" in EXAMPLES["te"]


def test_santali_and_sindhi_years_have_explicit_scale_words() -> None:
    assert "ᱢᱤᱫ ᱜᱮᱥᱟᱭ ᱤᱨᱟᱹᱞ ᱥᱟᱭ" in _rows("sat")["Date"][0]
    assert "ᱢᱤᱫ ᱜᱮᱥᱟᱭ ᱟᱨᱮ ᱥᱟᱭ" in _rows("sat")["Date"][0]
    assert "अरिड़हं सौ सतेतालीह" in _rows("sd")["Date"][0]
    assert "उणीह सौ नवे" in _rows("sd")["Date"][0]


@pytest.mark.parametrize(
    ("language", "category", "spoken", "written"),
    [
        (
            "hi",
            "Ordinal",
            "पहला / इक्कीसवाँ / पचासवाँ / इक्कीसवीं",
            "1वाँ / 21वाँ / 50वाँ / 21वीं",
        ),
        (
            "hi",
            "Money",
            "बावन डॉलर / दो सौ उनचास डॉलर और निन्यानवे सेंट / बावन रुपये चार पैसे",
            "$52 / $249.99 / ₹52.04",
        ),
        (
            "hi",
            "Duration",
            "एक घंटा / एक घंटा पाँच सेकंड / एक मिनट पाँच सेकंड",
            "1 घंटा / 1 घंटा 5 सेकंड / 1 मिनट 5 सेकंड",
        ),
        (
            "hi",
            "Time",
            "तीन बजकर पाँच मिनट पी एम / दस ए एम / एक बजकर पैंतालीस मिनट / दोपहर / चार बजे",
            "3:05 PM / 10:00 AM / 1:45 / दोपहर / 4:00",
        ),
        (
            "hi",
            "Fractions",
            "आधा / एक तिहाई / दो तिहाई / एक और तीन चौथाई / इक्कीस बटा दो",
            "1/2 / 1/3 / 2/3 / 1 3/4 / 21/2",
        ),
        (
            "mr",
            "Decimal digit sequence",
            "तीन पूर्णांक शून्य पाच",
            "3.05",
        ),
    ],
)
def test_approved_example_rows(language: str, category: str, spoken: str, written: str) -> None:
    assert _rows(language)[category] == (spoken, written)


@pytest.mark.parametrize(
    ("language", "approved_rule"),
    [
        (
            "hi",
            (
                "- In Hindi explicit measurement quantities, use these mappings: `मीटर` → "
                "`m`; `ग्राम` → `g`; `लीटर` → `L`; `डिग्री सेल्सियस` → `°C`; `वर्ग सेंटीमीटर` "
                "→ `cm²`. Keep one space between the newly written measurement amount and "
                "these units. These mappings do not license removing native inflections or "
                "rewriting unrelated wording."
            ),
        ),
        (
            "mr",
            (
                "- In Marathi clock expressions containing an hour and a minute count, `N "
                "वाजून M मिनिटे` means M minutes after hour N, while the clock pattern `N-ला "
                "M मिनिटे` means M minutes remaining until hour N. Subtract in the latter "
                "clock construction, including hour rollover; do not extend this reading to "
                "unrelated dative constructions."
            ),
        ),
    ],
)
def test_approved_language_specific_rules(language: str, approved_rule: str) -> None:
    assert EXAMPLES[language].count(approved_rule) == 1


@pytest.mark.parametrize(
    "approved_rule",
    [
        (
            "- Perform minimal-span editing: copy all text outside eligible conversion "
            "spans exactly as supplied, preserving wording, order, spelling, native "
            "grammatical endings, code-switching, script mixing, punctuation, whitespace "
            "and invisible Unicode characters. Preserve disfluencies, repetitions, false "
            "starts, colloquial forms, mispronunciations and grammatical errors. Do not "
            "correct spelling or grammar, normalize Unicode rendering or remove "
            "formatting characters outside an eligible conversion span."
        ),
        (
            "- Copy existing punctuation and whitespace exactly outside an eligible "
            "conversion span. Do not add, delete, replace or normalize punctuation or "
            "spacing merely to make the sentence look more conventional. Add or change "
            "punctuation only inside an explicitly licensed structured conversion, such "
            "as a newly converted decimal, clock, date, phone, currency or URL/email "
            "expression."
        ),
        (
            "Apply these conventions to every eligible newly converted expression across "
            "the transcript, including eligible spoken percent, measurement, currency or "
            "named-title forms beside an already-written numeral. A numeral already "
            "present as digits is not a spoken-number conversion span: preserve its "
            "digits, digit script, separators, leading zeroes and attached grammatical "
            "endings, even when another number in the same sentence is converted. "
            "Converting a neighboring spoken unit or percent word does not by itself "
            "authorize rewriting that numeral. Apply composite arithmetic or "
            "restructuring only when the applicable explicit currency rule "
            "requires it for that expression; this exception does not license normalizing "
            "unrelated written numbers or complete written dates and codes. Preserve "
            "unrelated identifiers, punctuation and script mixing; do not globally "
            "reformat the transcript or invent unreviewed symbol-plus-suffix forms."
        ),
        (
            "- Parse each complete cardinal expression as one numeric value before "
            "formatting it, using the active language's number vocabulary rather than a "
            "similarly sounding value. Combine its scale multipliers and lower-order "
            "remainder; consume every word belonging to the numeral, including an initial "
            "one that multiplies a scale word. Do not concatenate component values, drop "
            "a remainder or leave a multiplier word behind. Use ASCII digits for newly "
            "converted numbers. Group ordinary cardinal quantities and the integer part "
            "of currency amounts with Indian commas: the last group has three digits and "
            "preceding groups have two, as in `2,024`, `1,00,000`, and `1,23,456.78`. Do "
            "not group years, ordinal numbers, date/time fields, phone numbers, postal "
            "codes, house identifiers, fractions or structured letter-number codes."
        ),
        (
            "- When the input supplies a decimal marker, parse the complete integer part "
            "before the marker and join the explicitly spoken fractional digits in their "
            "spoken order after one decimal point. Preserve supplied leading and trailing "
            "fractional zeroes. Do not treat the fractional digits as separate "
            "quantities, merge them into the integer part or substitute a mixed-fraction "
            "representation."
        ),
        (
            "- Use the native ordinal suffix licensed by the actual spoken form, "
            "preserving its gender, case and attached grammatical endings. Do not replace "
            "a feminine or inflected ordinal with the masculine suffix merely because an "
            "example uses a masculine form. Do not introduce English `st`, `nd`, `rd` or "
            "`th`. Express decades with native suffixes or native decade words, not "
            "English `s`; retain a lexicalized native decade expression when no numeric "
            "suffix is established."
        ),
        (
            "- Determine the currency from the spoken currency unit and convert the "
            "complete eligible money expression, including its amount and currency-unit "
            "words; do not convert only its number words and leave the money expression "
            "partially spoken. Put currency symbols immediately before the amount and `%` "
            "immediately after the number: `$52`, `$249.99`, `₹1,00,000`, `3.5%`. Do not "
            "put a space between a money/percent symbol and its number. For currencies "
            "with 100 minor units per major unit, compute major amount plus minor amount "
            "divided by 100, carry excess minor units into the major amount and use two "
            "decimal places for an explicitly spoken minor amount. Thus 625 rupees and 2 "
            "paise is `₹625.02`. Preserve unrelated wording and grammatical endings "
            "outside the eligible money span."
        ),
        (
            "- Durations: for elapsed time rather than a clock reading, convert only "
            "the spoken numeric quantities to digits and retain the native duration-unit "
            "words, their grammatical endings, connectors, order and surrounding "
            "wording. Keep a fractional amount in its original unit using the "
            "active-language numeric form; do not convert hours to minutes or combine "
            "separate unit amounts. Do not render an elapsed duration as `H:MM` or "
            "`H:MM:SS`, pad quantities, insert missing fields or infer AM/PM. Genuine "
            "clock readings continue to follow the clock rules above."
        ),
        (
            "- When an active-language example or rule explicitly covers an ambiguous "
            "expression, follow its conversion pattern instead of the general defaults "
            "below. Examples are patterns, not fallback values: determine the value and "
            "expression type from the actual input. Do not add a currency symbol, decimal "
            "component, percent sign, ordinal suffix or decade suffix unless the input "
            "supplies the corresponding meaning."
        ),
        (
            "- For a spoken numerator-denominator expression, parse the complete "
            "numerator before the fraction separator and the complete denominator after "
            "it. Do not reinterpret the final word of a multiword numerator as a separate "
            "mixed-fraction numerator. A separate whole-number part requires an "
            "explicitly spoken mixed-fraction construction. Preserve the supplied "
            "numerator and denominator rather than simplifying the fraction or inventing "
            "a whole-number part."
        ),
        (
            "- Resolve the grammatical meaning of a number-like word in its actual "
            "sentence before converting it. Keep nonnumeric verbs and temporal or "
            "adverbial expressions in word form; for example, Hindi `पहले कभी` meaning "
            "ever before is not a first ordinal. Convert a homograph only where it "
            "actually denotes an eligible number or ordinal, preserving its native "
            "ending. Keep number expressions in word form when they are idiomatic, part "
            "of a proper noun, pronominal or indefinite, or a vague quantity, unless an "
            "active-language ordinal rule explicitly gives a suffix-preserving written "
            "form."
        ),
        (
            "- For an explicit measurement quantity, convert the numeral and its unit "
            "together when the active-language rules establish the unit's written symbol. "
            "Use the licensed unit spacing and exponent notation, such as `100 m`, `20 "
            "°C` and `56,030 cm²`. Do not leave the unit spoken after converting the "
            "numeral when that unit mapping is explicitly supplied. Preserve surrounding "
            "wording and native grammatical endings; do not invent a written form for an "
            "inflected or unfamiliar unit. Explicit elapsed-time expressions follow the "
            "duration rule rather than an unrelated measurement abbreviation."
        ),
        (
            "- In postal or ZIP codes and explicitly digit-spelled house identifiers, "
            "concatenate exactly one digit for each spoken digit word in order, retaining "
            "every repetition and zero. Do not drop repeated digits or insert unspoken "
            "digits. Convert cardinal/grouped house-number readings to their complete "
            "numeric value. Do not add grouping commas to identifiers or simplify an "
            "explicitly supplied slash/hyphen address identifier."
        ),
        (
            "- Render recognized acronyms and structured letter-number forms shown by the "
            "active-language examples in their conventional uppercase written form; do "
            "not expand them. Match the complete structured expression, not a "
            "letter-name-looking prefix inside an unrelated word or proper name. Do not "
            "split or transliterate an ordinary word merely because its beginning sounds "
            "like a letter name; preserve native endings outside a licensed code span."
        ),
        (
            "- Interpret active-language clock relations before formatting: a licensed "
            "minutes-past construction adds the minutes to the named hour, while a "
            "licensed minutes-to construction subtracts them from the upcoming named "
            "hour. Preserve the applicable 12-hour or explicitly supplied 24-hour "
            "convention, and do not infer AM/PM. Convert the complete eligible clock "
            "expression, including its internal clock-unit words, while preserving "
            "surrounding wording."
        ),
    ],
    ids=(
        "line-10-minimal-span-copying",
        "line-11-punctuation-and-spacing-boundaries",
        "line-21-written-numerals-inside-mixed-transcripts",
        "line-23-complete-cardinal-value-and-span",
        "after-line-23-decimal-digit-sequences",
        "line-24-ordinal-gender-and-endings",
        "line-25-complete-money-expressions",
        "line-31-elapsed-duration-preserves-native-units",
        "line-37-example-type-versus-input-meaning",
        "after-line-44-complete-numerator-versus-mixed-fraction",
        "line-42-nonnumeric-homographs-in-context",
        "after-line-25-complete-licensed-measurement-spans",
        "line-35-repeated-digits-in-addresses-and-postal-codes",
        "line-33-structured-code-word-boundaries",
        "after-line-27-clock-relations-and-complete-clock-spans",
    ),
)
def test_approved_common_prompt_rules(approved_rule: str) -> None:
    assert PROMPT.count(approved_rule) == 1
