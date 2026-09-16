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

import argparse
import importlib.util
import json
import re
import unicodedata
from pathlib import Path

import pytest

from nemo_curator.stages.audio.text_filtering.pnc_language_rules import PNC_LANGUAGE_CODES
from nemo_curator.stages.audio.text_filtering.remote_text_llm_stage import RemoteTextLLMStage
from nemo_curator.stages.audio.text_filtering.text_llm_stage import TextLLMStage
from nemo_curator.stages.audio.text_filtering.tn_language_examples import (
    TN_LANGUAGE_CODES,
    load_tn_language_examples,
)

PROMPT_DIR = Path(__file__).parents[4] / "nemo_curator" / "stages" / "audio" / "text_filtering" / "prompts"
RUN_TEXT_PIPELINE = Path(__file__).parents[4] / "examples" / "audio" / "text_processing" / "run_text_pipeline.py"

_RUNNER_SPEC = importlib.util.spec_from_file_location("curator_run_text_pipeline", RUN_TEXT_PIPELINE)
if _RUNNER_SPEC is None or _RUNNER_SPEC.loader is None:
    message = f"Unable to load {RUN_TEXT_PIPELINE}"
    raise RuntimeError(message)
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)

EXPECTED_CATEGORIES = (
    "Cardinal",
    "Ordinal",
    "Date",
    "Time",
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
EXPECTED_LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi (Gurmukhi)",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu (Perso-Arabic)",
    "brx": "Bodo (Devanagari)",
    "doi": "Dogri (Devanagari)",
    "kok": "Konkani (Devanagari)",
    "ks": "Kashmiri (Perso-Arabic)",
    "mai": "Maithili",
    "mni": "Manipuri (Meetei Mayek)",
    "ne": "Nepali",
    "sa": "Sanskrit (Devanagari)",
    "sat": "Santali (Ol Chiki)",
    "sd": "Sindhi (Devanagari)",
}
EXPECTED_EXAMPLE_DELIMITERS = {
    "Cardinal": 2,
    "Ordinal": 2,
    "Date": 1,
    "Time": 3,
    "Money": 1,
    "Percent": 1,
    "Units": 2,
    "Fractions": 3,
    "Phone": 1,
    "URL/Email": 1,
    "Titles": 3,
    "Address": 0,
    "Roman num.": 1,
    "Negative": 1,
    "Decades": 1,
    "Letter+num": 1,
}
EXPECTED_INVARIANT_WRITTEN_EXAMPLES = {
    "Cardinal": "14 / 1,030.5 / 2024",
    "Money": "$52 / $249.99",
    "Units": "5 kg / 90 km/h / 5'4\"",
    "Fractions": "1/2 / 1/3 / 2/3 / 1 3/4",
    "Phone": "5558675309 / 18005550199",
    "URL/Email": "example.com/pricing / john@gmail.com",
    "Decades": "70s / 20s",
    "Letter+num": "Q2 / B12",
}
EXPECTED_SCRIPT_NAMES = {
    "as": "BENGALI",
    "bn": "BENGALI",
    "gu": "GUJARATI",
    "hi": "DEVANAGARI",
    "kn": "KANNADA",
    "ml": "MALAYALAM",
    "mr": "DEVANAGARI",
    "or": "ORIYA",
    "pa": "GURMUKHI",
    "ta": "TAMIL",
    "te": "TELUGU",
    "ur": "ARABIC",
    "brx": "DEVANAGARI",
    "doi": "DEVANAGARI",
    "kok": "DEVANAGARI",
    "ks": "ARABIC",
    "mai": "DEVANAGARI",
    "mni": "MEETEI MAYEK",
    "ne": "DEVANAGARI",
    "sa": "DEVANAGARI",
    "sat": "OL CHIKI",
    "sd": "DEVANAGARI",
}
INDIC_SCRIPT_NAMES = frozenset(EXPECTED_SCRIPT_NAMES.values())


def _parse_tn_cli_args(*extra_args: str) -> argparse.Namespace:
    return _RUNNER._build_arg_parser().parse_args(
        ["--input_manifest", "input.jsonl", "--output_dir", "output", *extra_args]
    )


def _table_rows(fragment: str) -> dict[str, tuple[str, str]]:
    table_lines = [line for line in fragment.splitlines() if line.startswith("|")]
    assert table_lines[:2] == [
        "| Category | Written | Spoken |",
        "|---|---|---|",
    ]
    rows: dict[str, tuple[str, str]] = {}
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 3
        assert all(cells)
        category, written, spoken = cells
        assert category not in rows
        rows[category] = (written, spoken)
    assert len(table_lines) == 18
    return rows


def test_tn_prompt_cli_defaults_to_shared_prompt() -> None:
    args = _parse_tn_cli_args()

    assert _RUNNER._resolve_tn_prompt_file(
        args.tn_prompt_file,
        use_indic_prompt=args.use_indic_tn_prompt,
    ) == str(_RUNNER._TN_PROMPT)


def test_tn_prompt_cli_selects_bundled_indic_prompt() -> None:
    args = _parse_tn_cli_args("--use_indic_tn_prompt")

    assert _RUNNER._resolve_tn_prompt_file(
        args.tn_prompt_file,
        use_indic_prompt=args.use_indic_tn_prompt,
    ) == str(_RUNNER._TN_INDIC_PROMPT)


def test_tn_prompt_cli_preserves_custom_prompt_path() -> None:
    args = _parse_tn_cli_args("--tn_prompt_file", "custom-tn.md")

    assert (
        _RUNNER._resolve_tn_prompt_file(
            args.tn_prompt_file,
            use_indic_prompt=args.use_indic_tn_prompt,
        )
        == "custom-tn.md"
    )


def test_tn_prompt_cli_rejects_bundled_and_custom_prompt_together() -> None:
    with pytest.raises(SystemExit):
        _parse_tn_cli_args(
            "--use_indic_tn_prompt",
            "--tn_prompt_file",
            "custom-tn.md",
        )


def test_tn_prompt_cli_accepts_custom_language_examples_path() -> None:
    args = _parse_tn_cli_args("--tn_language_examples_file", "custom-examples.json")

    assert args.tn_language_examples_file == "custom-examples.json"


def test_tn_language_examples_load_only_for_enabled_placeholder_prompt(tmp_path: Path) -> None:
    assert (
        _RUNNER._load_tn_language_examples_for_prompt(
            enabled=False,
            prompt_file="missing-prompt.md",
            examples_file="missing-examples.json",
        )
        is None
    )
    assert (
        _RUNNER._load_tn_language_examples_for_prompt(
            enabled=True,
            prompt_file=str(PROMPT_DIR / "tn_prompt.md"),
            examples_file="missing-examples.json",
        )
        is None
    )

    custom_prompt = tmp_path / "custom-tn.md"
    custom_prompt.write_text("{language_rules}\n{text}", encoding="utf-8")
    custom_examples = tmp_path / "custom-examples.json"
    custom_examples.write_text(json.dumps(load_tn_language_examples(), ensure_ascii=False), encoding="utf-8")

    assert (
        _RUNNER._load_tn_language_examples_for_prompt(
            enabled=True,
            prompt_file=str(custom_prompt),
            examples_file=str(custom_examples),
        )
        == load_tn_language_examples()
    )


def test_bundled_language_examples_have_exact_target_codes_and_categories() -> None:
    examples = load_tn_language_examples()

    assert tuple(examples) == TN_LANGUAGE_CODES
    assert TN_LANGUAGE_CODES == PNC_LANGUAGE_CODES
    for code, fragment in examples.items():
        assert fragment.splitlines()[0] == f"### {EXPECTED_LANGUAGE_NAMES[code]} (`{code}`) conversion examples"
        assert tuple(_table_rows(fragment)) == EXPECTED_CATEGORIES
        assert "Spoken structural forms:" in fragment


def test_bundled_examples_preserve_source_example_arity_and_code_switching() -> None:
    examples = load_tn_language_examples()

    for fragment in examples.values():
        rows = _table_rows(fragment)
        for category, expected_delimiters in EXPECTED_EXAMPLE_DELIMITERS.items():
            written, spoken = rows[category]
            assert written.count(" / ") == expected_delimiters
            assert spoken.count(" / ") == expected_delimiters

        for category, expected_written in EXPECTED_INVARIANT_WRITTEN_EXAMPLES.items():
            assert rows[category][0] == expected_written
        assert rows["Time"][0].startswith("3:05 PM / 10 AM / 1:45 / ")

        url_written, url_spoken = rows["URL/Email"]
        assert url_written == "example.com/pricing / john@gmail.com"
        assert re.findall(r"[a-z]+", url_spoken) == ["example", "com", "pricing", "john", "gmail", "com"]


def test_reported_language_corrections_are_applied() -> None:
    examples = load_tn_language_examples()

    assert _table_rows(examples["gu"])["Ordinal"][0] == "1લું / 21મું / 50મું"
    assert "उनान्सय सेन्ट" in _table_rows(examples["ne"])["Money"][1]

    punjabi_money = _table_rows(examples["pa"])["Money"][1]
    assert "ਉਣਿੰਜਾ ਡਾਲਰ" in punjabi_money
    assert "ਨੜਿੰਨਵੇਂ ਸੈਂਟ" in punjabi_money

    assert _table_rows(examples["doi"])["Ordinal"][0] == "1मां / 21मां / 50मां"

    kashmiri = examples["ks"]
    assert "پَنٛژٲہیُٛم" in kashmiri
    assert "سَتہٕ تٲجی" in kashmiri
    assert kashmiri.count("پؠٹھٕ") == 2
    for stale_form in ("پَنٛژاہُیٛم", "سَتتٲجی", "پیٹھٕ"):
        assert stale_form not in kashmiri

    maithili_titles = _table_rows(examples["mai"])["Titles"][1]
    assert "प्रोफेसर जोन्स" in maithili_titles
    assert "अध्यापक जोन्स" not in maithili_titles

    manipuri = examples["mni"]
    assert manipuri.count("ꯅꯤꯝꯐꯨ") == 3
    assert manipuri.count("ꯊꯣꯢ") == 6
    assert "ꯅꯤꯐꯨ" not in manipuri
    assert "ꯊꯣꯏ" not in manipuri

    santali_cardinals = _table_rows(examples["sat"])["Cardinal"][1].split(" / ")
    assert santali_cardinals[2] == "ᱵᱟᱨ ᱦᱟᱡᱟᱨ ᱵᱟᱨ ᱜᱮᱞ ᱯᱩᱱ"

    odia_times = _table_rows(examples["or"])["Time"][1].split(" / ")
    assert odia_times[0] == "ତିନିଟା ପାଞ୍ଚ ମିନିଟ୍ ପି ଏମ୍"
    assert odia_times[2] == "ଏକଟା ପଞ୍ଚଚାଳିଶ ମିନିଟ୍"

    telugu_regnal = _table_rows(examples["te"])["Roman num."][1].split(" / ")
    assert telugu_regnal[0] == "రాజు హెన్రీ ఎనిమిదవ"


def test_every_spoken_example_uses_its_configured_script() -> None:
    examples = load_tn_language_examples()

    for code, fragment in examples.items():
        script_name = EXPECTED_SCRIPT_NAMES[code]
        for category, (_, spoken) in _table_rows(fragment).items():
            assert any(script_name in unicodedata.name(character, "") for character in spoken), (code, category)
            for character in spoken:
                character_name = unicodedata.name(character, "")
                detected_scripts = {name for name in INDIC_SCRIPT_NAMES if name in character_name}
                assert not detected_scripts or detected_scripts == {script_name}, (code, category, character)


def test_language_examples_loader_rejects_wrong_code_order(tmp_path: Path) -> None:
    examples = load_tn_language_examples()
    reversed_path = tmp_path / "reversed.json"
    reversed_path.write_text(json.dumps(dict(reversed(examples.items())), ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected TN language codes in order"):
        load_tn_language_examples(reversed_path)


@pytest.mark.parametrize("bad_value", [None, "", "   ", [], {}])
def test_language_examples_loader_rejects_empty_or_non_string_fragments(tmp_path: Path, bad_value: object) -> None:
    examples: dict[str, object] = load_tn_language_examples()
    examples["hi"] = bad_value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a non-empty string"):
        load_tn_language_examples(path)


def test_indic_tn_prompt_uses_one_language_rules_placeholder() -> None:
    prompt = (PROMPT_DIR / "tn_prompt_indic.md").read_text(encoding="utf-8")

    assert prompt.count("{language_rules}") == 1
    assert "natural cardinal or grouped-number reading" in prompt
    assert "zip/house numbers digit by digit" not in prompt
    for source_example in ("fourteen / one thousand", "um, uh", "go- going", "gonna", "oh eleven", "One Direction"):
        assert source_example not in prompt


def test_shared_tn_prompt_remains_language_general() -> None:
    prompt = (PROMPT_DIR / "tn_prompt.md").read_text(encoding="utf-8")

    assert "{language_rules}" not in prompt
    assert "fourteen / one thousand" in prompt
    assert "natural cardinal or grouped-number readings" in prompt
    assert "zip/house numbers → digit-by-digit" not in prompt


def test_text_stage_renders_only_active_language_examples() -> None:
    stage = TextLLMStage(
        prompt_text="{language}\n{language_rules}\n{text}",
        language_rules={"hi": "Hindi examples.", "ur": "Urdu examples."},
    )
    stage._system_prompt = stage._resolve_prompt()

    rendered = stage._render_prompt_template("नमस्ते दुनिया", {"source_lang": "hi"})

    assert rendered == "hi\nHindi examples.\nनमस्ते दुनिया"
    assert "Urdu examples." not in rendered


def test_remote_stage_uses_same_row_scoped_rendering() -> None:
    stage = RemoteTextLLMStage(
        prompt_text="{language}\n{language_rules}\n{text}",
        language_rules={"hi": "Hindi examples.", "ur": "Urdu examples."},
    )
    stage._system_prompt = stage._resolve_prompt()

    messages = stage._build_messages("नमस्ते दुनिया", {"source_lang": "hi"})

    assert messages == [{"role": "user", "content": "hi\nHindi examples.\nनमस्ते दुनिया"}]


@pytest.mark.parametrize("task_data", [None, {}, {"source_lang": ""}, {"source_lang": "xx"}])
def test_tn_language_example_resolution_fails_closed(task_data: dict | None) -> None:
    stage = TextLLMStage(
        prompt_text="{language_rules}\n{text}",
        language_rules={"hi": "Hindi examples."},
    )
    stage._system_prompt = stage._resolve_prompt()

    with pytest.raises(ValueError, match=r"prompt requires|unsupported"):
        stage._render_prompt_template("नमस्ते दुनिया", task_data)


def test_prompt_without_language_examples_remains_backward_compatible() -> None:
    stage = TextLLMStage(prompt_text="{language}\n{text}")
    stage._system_prompt = stage._resolve_prompt()

    assert stage._render_prompt_template("hello", None) == "English\nhello"


@pytest.mark.parametrize("language", TN_LANGUAGE_CODES)
def test_bundled_render_contains_only_active_language_examples(language: str) -> None:
    examples = load_tn_language_examples()
    stage = TextLLMStage(
        prompt_file=str(PROMPT_DIR / "tn_prompt_indic.md"),
        language_rules=examples,
    )
    stage._system_prompt = stage._resolve_prompt()

    rendered = stage._render_prompt_template("sample 14", {"source_lang": language})

    assert examples[language] in rendered
    for other_code, other_fragment in examples.items():
        if other_code != language:
            heading = other_fragment.splitlines()[0]
            assert heading not in rendered
