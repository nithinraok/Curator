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
import sys
import unicodedata
from pathlib import Path

import pytest

from nemo_curator.stages.audio.text_filtering.itn_language_examples import (
    ITN_LANGUAGE_CODES,
    load_itn_language_examples,
)
from nemo_curator.stages.audio.text_filtering.pnc_language_rules import PNC_LANGUAGE_CODES
from nemo_curator.stages.audio.text_filtering.remote_text_llm_stage import RemoteTextLLMStage
from nemo_curator.stages.audio.text_filtering.text_llm_stage import TextLLMStage

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
    "Duration": 0,
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
    "Cardinal": "14 / 1,030.5 / 2,024",
    "Money": "$52 / $249.99",
    "Units": "5 kg / 90 km/h / 5'4\"",
    "Fractions": "1/2 / 1/3 / 2/3 / 1 3/4",
    "Phone": "5558675309 / 18005550199",
    "URL/Email": "example.com/pricing / john@gmail.com",
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
STRUCTURAL_TARGETS = (".", "@", "/", ":", "-", "0")


def _parse_itn_cli_args(*extra_args: str) -> argparse.Namespace:
    return _RUNNER._build_arg_parser().parse_args(
        ["--input_manifest", "input.jsonl", "--output_dir", "output", *extra_args]
    )


def _table_rows(fragment: str) -> dict[str, tuple[str, str]]:
    table_lines = [line for line in fragment.splitlines() if line.startswith("|")]
    assert table_lines[:2] == [
        "| Category | Spoken | Written |",
        "|---|---|---|",
    ]
    rows: dict[str, tuple[str, str]] = {}
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 3
        assert all(cells)
        category, spoken, written = cells
        assert category not in rows
        rows[category] = (spoken, written)
    return rows


def test_itn_prompt_cli_defaults_to_shared_prompt() -> None:
    args = _parse_itn_cli_args()

    assert _RUNNER._resolve_itn_prompt_file(
        args.itn_prompt_file,
        use_indic_prompt=args.use_indic_itn_prompt,
    ) == str(_RUNNER._ITN_PROMPT)


def test_itn_prompt_cli_selects_bundled_indic_prompt() -> None:
    args = _parse_itn_cli_args("--use_indic_itn_prompt")

    assert _RUNNER._resolve_itn_prompt_file(
        args.itn_prompt_file,
        use_indic_prompt=args.use_indic_itn_prompt,
    ) == str(_RUNNER._ITN_INDIC_PROMPT)


def test_itn_prompt_cli_preserves_custom_prompt_path() -> None:
    args = _parse_itn_cli_args("--itn_prompt_file", "custom-itn.md")

    assert (
        _RUNNER._resolve_itn_prompt_file(
            args.itn_prompt_file,
            use_indic_prompt=args.use_indic_itn_prompt,
        )
        == "custom-itn.md"
    )


def test_itn_prompt_cli_rejects_bundled_and_custom_prompt_together() -> None:
    with pytest.raises(SystemExit):
        _parse_itn_cli_args(
            "--use_indic_itn_prompt",
            "--itn_prompt_file",
            "custom-itn.md",
        )


def test_itn_prompt_cli_accepts_custom_language_examples_path() -> None:
    args = _parse_itn_cli_args("--itn_language_examples_file", "custom-examples.json")

    assert args.itn_language_examples_file == "custom-examples.json"


def test_itn_language_examples_load_only_for_enabled_placeholder_prompt(tmp_path: Path) -> None:
    assert (
        _RUNNER._load_itn_language_examples_for_prompt(
            enabled=False,
            prompt_file="missing-prompt.md",
            examples_file="missing-examples.json",
        )
        is None
    )
    assert (
        _RUNNER._load_itn_language_examples_for_prompt(
            enabled=True,
            prompt_file=str(PROMPT_DIR / "itn_prompt.md"),
            examples_file="missing-examples.json",
        )
        is None
    )

    custom_prompt = tmp_path / "custom-itn.md"
    custom_prompt.write_text("{language_rules}\n{text}", encoding="utf-8")
    custom_examples = tmp_path / "custom-examples.json"
    custom_examples.write_text(json.dumps(load_itn_language_examples(), ensure_ascii=False), encoding="utf-8")

    assert (
        _RUNNER._load_itn_language_examples_for_prompt(
            enabled=True,
            prompt_file=str(custom_prompt),
            examples_file=str(custom_examples),
        )
        == load_itn_language_examples()
    )


@pytest.mark.parametrize("enable_flag", ["--enable_itn", "--enable_itn_no-disfluencies"])
def test_each_itn_enable_flag_loads_and_wires_row_scoped_examples(
    enable_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = {"hi": "Hindi examples."}
    load_call: dict[str, object] = {}
    stage_call: dict[str, object] = {}

    class StopAfterITNStageError(RuntimeError):
        pass

    def fake_load(**kwargs: object) -> dict[str, str]:
        load_call.update(kwargs)
        return examples

    def fake_text_stage(**kwargs: object) -> None:
        if kwargs.get("name") == "ITNRestoration":
            stage_call.update(kwargs)
            raise StopAfterITNStageError

    monkeypatch.setattr(_RUNNER, "_load_itn_language_examples_for_prompt", fake_load)
    monkeypatch.setattr(_RUNNER, "TextLLMStage", fake_text_stage)
    monkeypatch.setattr(_RUNNER, "ALMManifestReader", lambda **_kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUN_TEXT_PIPELINE),
            "--input_manifest",
            "input.jsonl",
            "--output_dir",
            "output",
            "--use_indic_itn_prompt",
            enable_flag,
        ],
    )

    with pytest.raises(StopAfterITNStageError):
        _RUNNER.main()

    assert load_call == {
        "enabled": True,
        "prompt_file": str(_RUNNER._ITN_INDIC_PROMPT),
        "examples_file": str(_RUNNER._ITN_LANGUAGE_EXAMPLES),
    }
    assert stage_call["prompt_file"] == str(_RUNNER._ITN_INDIC_PROMPT)
    assert stage_call["language_rules"] is examples
    assert stage_call["enable_validation"] is False


@pytest.mark.parametrize("use_custom_prompt", [False, True])
def test_unmapped_itn_prompt_keeps_generic_validation_enabled(
    use_custom_prompt: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_call: dict[str, object] = {}

    class StopAfterITNStageError(RuntimeError):
        pass

    def fake_text_stage(**kwargs: object) -> None:
        if kwargs.get("name") == "ITNRestoration":
            stage_call.update(kwargs)
            raise StopAfterITNStageError

    prompt_args: list[str] = []
    if use_custom_prompt:
        custom_prompt = tmp_path / "custom-itn.md"
        custom_prompt.write_text("Normalize this transcript:\n{text}\n", encoding="utf-8")
        prompt_args = ["--itn_prompt_file", str(custom_prompt)]

    monkeypatch.setattr(_RUNNER, "TextLLMStage", fake_text_stage)
    monkeypatch.setattr(_RUNNER, "ALMManifestReader", lambda **_kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUN_TEXT_PIPELINE),
            "--input_manifest",
            "input.jsonl",
            "--output_dir",
            "output",
            "--enable_itn",
            "--itn_language_examples_file",
            "missing-examples.json",
            *prompt_args,
        ],
    )

    with pytest.raises(StopAfterITNStageError):
        _RUNNER.main()

    assert stage_call["language_rules"] is None
    assert stage_call["enable_validation"] is True


@pytest.mark.parametrize(
    ("fused_stage", "post_fused_stages", "expected"),
    [
        (None, ["ITNRestoration"], ["TextNormalization", "ITNRestoration"]),
        (
            None,
            ["ITNRestoration", "DisfluencyRemoval"],
            ["TextNormalization", "ITNRestoration", "DisfluencyRemoval"],
        ),
        (
            "FusedRemoteTextLLMStage",
            ["DisfluencyRemoval"],
            ["TextNormalization", "FusedRemoteTextLLMStage", "DisfluencyRemoval"],
        ),
        (None, [], ["TextNormalization"]),
    ],
    ids=("tn-itn", "tn-itn-disfluency", "fused-then-disfluency", "no-fused-work"),
)
def test_fused_stage_sequence_preserves_dependent_stages(
    fused_stage: object | None,
    post_fused_stages: list,
    expected: list,
) -> None:
    stages = ["TextNormalization"]

    _RUNNER._append_fused_stage_sequence(
        stages,
        fused_stage=fused_stage,
        post_fused_stages=post_fused_stages,
    )

    assert stages == expected


def test_bundled_language_examples_have_exact_target_codes_and_categories() -> None:
    examples = load_itn_language_examples()

    assert tuple(examples) == ITN_LANGUAGE_CODES
    assert ITN_LANGUAGE_CODES == PNC_LANGUAGE_CODES
    for code, fragment in examples.items():
        assert fragment.splitlines()[0] == f"### {EXPECTED_LANGUAGE_NAMES[code]} (`{code}`) conversion examples"
        expected_categories = EXPECTED_CATEGORIES
        if code == "mr":
            expected_categories = (*EXPECTED_CATEGORIES[:1], "Decimal digit sequence", *EXPECTED_CATEGORIES[1:])
        assert tuple(_table_rows(fragment)) == expected_categories
        footer = next(
            line for line in fragment.splitlines() if line.startswith("Structural forms (spoken → written):")
        )
        assert tuple(re.findall(r"→ `([^`]+)`", footer)) == STRUCTURAL_TARGETS


def test_bundled_examples_preserve_source_example_arity_and_written_forms() -> None:
    examples = load_itn_language_examples()

    for language, fragment in examples.items():
        rows = _table_rows(fragment)
        expected_delimiters_by_category = dict(EXPECTED_EXAMPLE_DELIMITERS)
        expected_written_by_category = dict(EXPECTED_INVARIANT_WRITTEN_EXAMPLES)
        if language == "hi":
            expected_delimiters_by_category.update(
                {"Ordinal": 3, "Time": 4, "Duration": 2, "Money": 2, "Fractions": 4}
            )
            expected_written_by_category.update(
                Money="$52 / $249.99 / ₹52.04",
                Fractions="1/2 / 1/3 / 2/3 / 1 3/4 / 21/2",
            )
        elif language == "mr":
            expected_delimiters_by_category["Decimal digit sequence"] = 0
            expected_written_by_category["Decimal digit sequence"] = "3.05"
        for category, expected_delimiters in expected_delimiters_by_category.items():
            spoken, written = rows[category]
            assert spoken.count(" / ") == expected_delimiters
            assert written.count(" / ") == expected_delimiters

        for category, expected_written in expected_written_by_category.items():
            assert rows[category][1] == expected_written
        assert rows["Time"][1].startswith("3:05 PM / 10:00 AM / 1:45 / ")
        duration_words = rows["Duration"][0].split(" ")
        duration_words[-1 if language == "mni" else 0] = "1"
        expected_duration = (
            "1 घंटा / 1 घंटा 5 सेकंड / 1 मिनट 5 सेकंड" if language == "hi" else " ".join(duration_words)
        )
        assert rows["Duration"][1] == expected_duration

        url_spoken, url_written = rows["URL/Email"]
        assert url_written == "example.com/pricing / john@gmail.com"
        assert re.findall(r"[a-z]+", url_spoken) == ["example", "com", "pricing", "john", "gmail", "com"]


def test_telugu_itn_rules_cover_reviewed_edge_cases() -> None:
    telugu_examples = load_itn_language_examples()["te"]
    rows = _table_rows(telugu_examples)

    assert rows["Ordinal"] == (
        "మొదటిది / ఇరవై ఒకటవది / యాభైవది",
        "1వది / 21వది / 50వది",
    )
    for reviewed_rule in (
        'పన్నెండు ఇంచెస్ → `12"`',
        "పన్నెండు బై పన్నెండు → `12x12`",
        "అతను డాక్టర్ → `అతను డాక్టర్`",
        "డాక్టర్గా → `డాక్టర్గా`",
        "డాక్టర్లు → `డాక్టర్లు`",
        "మొదటిది → `1వది`",
        "మొదటిసార → `1వసారి`",
        "రెండులోని → `2లోని`",
        "తొమ్మిది పది → `9:10`",
        "తొమ్మిది ముప్పై → `9:30`",
        "ఇద్దరు పిల్లలకి → `ఇద్దరు పిల్లలకి`",
        "ముప్పావు కప్పు → `3/4 కప్పు`",
        "ఒకటిన్నర గంటలు → `1.5 గంటలు`",
    ):
        assert reviewed_rule in telugu_examples


def test_synced_reviewed_language_corrections_are_retained() -> None:
    examples = load_itn_language_examples()
    reviewed_snippets = {
        "gu": ("1લું / 21મું / 50મું",),
        "ne": ("उनान्सय सेन्ट",),
        "pa": ("ਉਣਿੰਜਾ ਡਾਲਰ", "ਨੜਿੰਨਵੇਂ ਸੈਂਟ"),
        "doi": ("1मां / 21मां / 50मां",),
        "ks": ("پَنٛژٲہیُٛم", "سَتہٕ تٲجی", "پؠٹھٕ"),
        "mai": ("प्रोफेसर जोन्स",),
        "mni": ("ꯅꯤꯝꯐꯨ", "ꯅꯤꯊꯣꯢ"),
        "sat": ("ᱵᱟᱨ ᱦᱟᱡᱟᱨ ᱵᱟᱨ ᱜᱮᱞ ᱯᱩᱱ",),
        "or": ("ତିନିଟା ପାଞ୍ଚ ମିନିଟ୍", "ଏକଟା ପଞ୍ଚଚାଳିଶ ମିନିଟ୍"),
    }

    for language, snippets in reviewed_snippets.items():
        for snippet in snippets:
            assert snippet in examples[language]


def test_every_spoken_example_uses_its_configured_script() -> None:
    examples = load_itn_language_examples()

    for code, fragment in examples.items():
        script_name = EXPECTED_SCRIPT_NAMES[code]
        for category, (spoken, _) in _table_rows(fragment).items():
            assert any(script_name in unicodedata.name(character, "") for character in spoken), (code, category)
            for character in spoken:
                character_name = unicodedata.name(character, "")
                detected_scripts = {name for name in INDIC_SCRIPT_NAMES if name in character_name}
                assert not detected_scripts or detected_scripts == {script_name}, (code, category, character)


def test_language_examples_loader_rejects_non_object_schema(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="ITN language examples must be a JSON object"):
        load_itn_language_examples(path)


def test_language_examples_loader_rejects_wrong_code_order(tmp_path: Path) -> None:
    examples = load_itn_language_examples()
    reversed_path = tmp_path / "reversed.json"
    reversed_path.write_text(json.dumps(dict(reversed(examples.items())), ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected ITN language codes in order"):
        load_itn_language_examples(reversed_path)


@pytest.mark.parametrize("bad_value", [None, "", "   ", [], {}])
def test_language_examples_loader_rejects_empty_or_non_string_fragments(tmp_path: Path, bad_value: object) -> None:
    examples: dict[str, object] = load_itn_language_examples()
    examples["hi"] = bad_value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a non-empty string"):
        load_itn_language_examples(path)


def test_indic_itn_prompt_uses_one_language_rules_placeholder_and_preserves_controls() -> None:
    prompt = (PROMPT_DIR / "itn_prompt_indic.md").read_text(encoding="utf-8")

    assert prompt.count("{language_rules}") == 1
    assert "{language}" in prompt
    assert "{text}" in prompt
    assert "listed spoken forms" in prompt
    for required_control in (
        "disfluencies",
        "repetitions",
        "false starts",
        "colloquial forms",
        "mispronunciations",
        "grammatical errors",
        "proper noun",
        "pronominal or indefinite",
        "vague quantity",
        "temporal or financial",
        "true fraction",
        "final clean numeric expression",
        "directly modifies a person's name",
        "active-language example or rule explicitly covers",
        "lexical human-count form",
        "suffix-preserving written form",
    ):
        assert required_control in prompt
    for source_example in (
        "fourteen / one thousand",
        "um, uh",
        "go- going",
        "gonna",
        "NASA",
        "One Direction",
        "fourth quarter",
        "half asleep",
        "o- o- one hundred",
    ):
        assert source_example not in prompt


def test_shared_itn_prompt_remains_unchanged() -> None:
    prompt = (PROMPT_DIR / "itn_prompt.md").read_text(encoding="utf-8")

    assert "{language_rules}" not in prompt
    assert "fourteen / one thousand thirty point five" in prompt
    assert '"One Direction" (proper noun)' in prompt
    assert '"o- o- one hundred" → "o- o- 100"' in prompt


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
def test_itn_language_example_resolution_fails_closed(task_data: dict | None) -> None:
    stage = TextLLMStage(
        prompt_text="{language_rules}\n{text}",
        language_rules={"hi": "Hindi examples."},
    )
    stage._system_prompt = stage._resolve_prompt()

    with pytest.raises(ValueError, match=r"prompt requires|unsupported"):
        stage._render_prompt_template("नमस्ते दुनिया", task_data)


def test_itn_language_example_resolution_fails_without_mapping() -> None:
    stage = TextLLMStage(prompt_text="{language_rules}\n{text}")
    stage._system_prompt = stage._resolve_prompt()

    with pytest.raises(ValueError, match="no mapping was configured"):
        stage._render_prompt_template("नमस्ते दुनिया", {"source_lang": "hi"})


def test_prompt_without_language_examples_remains_backward_compatible() -> None:
    stage = TextLLMStage(prompt_text="{language}\n{text}")
    stage._system_prompt = stage._resolve_prompt()

    assert stage._render_prompt_template("hello", None) == "English\nhello"


@pytest.mark.parametrize("language", ITN_LANGUAGE_CODES)
def test_bundled_render_contains_only_active_language_examples(language: str) -> None:
    examples = load_itn_language_examples()
    stage = TextLLMStage(
        prompt_file=str(PROMPT_DIR / "itn_prompt_indic.md"),
        language_rules=examples,
    )
    stage._system_prompt = stage._resolve_prompt()

    rendered = stage._render_prompt_template("sample fourteen", {"source_lang": language})

    assert examples[language] in rendered
    for other_code, other_fragment in examples.items():
        if other_code != language:
            heading = other_fragment.splitlines()[0]
            assert heading not in rendered
