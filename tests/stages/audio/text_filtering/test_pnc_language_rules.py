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
from pathlib import Path

import pytest

from nemo_curator.stages.audio.text_filtering.pnc_language_rules import (
    PNC_LANGUAGE_CODES,
    load_pnc_language_rules,
)
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

ARABIC_LANGUAGE_CODES = {"ks", "ur"}
NON_ARABIC_PUNCTUATION = (".", "?", "!", ",", ";", ":", "...")
ARABIC_PUNCTUATION = ("\u06d4", "\u061f", "!", "\u060c", "\u061b", ":", "…")


def _parse_pnc_cli_args(*extra_args: str) -> argparse.Namespace:
    return _RUNNER._build_arg_parser().parse_args(
        ["--input_manifest", "input.jsonl", "--output_dir", "output", *extra_args]
    )


def test_pnc_prompt_cli_defaults_to_shared_prompt() -> None:
    args = _parse_pnc_cli_args()

    assert _RUNNER._resolve_pnc_prompt_file(
        args.pnc_prompt_file,
        use_indic_prompt=args.use_indic_pnc_prompt,
    ) == str(_RUNNER._PNC_PROMPT)


def test_pnc_prompt_cli_selects_bundled_indic_prompt() -> None:
    args = _parse_pnc_cli_args("--use_indic_pnc_prompt")

    assert _RUNNER._resolve_pnc_prompt_file(
        args.pnc_prompt_file,
        use_indic_prompt=args.use_indic_pnc_prompt,
    ) == str(_RUNNER._PNC_INDIC_PROMPT)


def test_pnc_prompt_cli_preserves_custom_prompt_path() -> None:
    args = _parse_pnc_cli_args("--pnc_prompt_file", "custom-pnc.md")

    assert (
        _RUNNER._resolve_pnc_prompt_file(
            args.pnc_prompt_file,
            use_indic_prompt=args.use_indic_pnc_prompt,
        )
        == "custom-pnc.md"
    )


def test_pnc_prompt_cli_rejects_bundled_and_custom_prompt_together() -> None:
    with pytest.raises(SystemExit):
        _parse_pnc_cli_args(
            "--use_indic_pnc_prompt",
            "--pnc_prompt_file",
            "custom-pnc.md",
        )


def test_bundled_language_rules_have_exact_target_codes() -> None:
    rules = load_pnc_language_rules()

    assert tuple(rules) == PNC_LANGUAGE_CODES
    assert all(rule.strip() for rule in rules.values())


def test_bundled_language_rules_follow_script_punctuation_policy() -> None:
    rules = load_pnc_language_rules()

    for language, rule in rules.items():
        punctuation = tuple(rule.split("`")[1::2])
        expected = ARABIC_PUNCTUATION if language in ARABIC_LANGUAGE_CODES else NON_ARABIC_PUNCTUATION
        assert punctuation == expected

    assert "ellipsis sequence `...`" in rules["mni"]
    assert "ellipsis `…`" in rules["ks"]


def test_indic_pnc_prompt_uses_one_language_rules_placeholder() -> None:
    prompt = (PROMPT_DIR / "pnc_prompt_indic.md").read_text(encoding="utf-8")

    assert prompt.count("{language_rules}") == 1
    assert "For Assamese, Bengali" not in prompt


def test_shared_pnc_prompt_remains_language_general() -> None:
    prompt = (PROMPT_DIR / "pnc_prompt.md").read_text(encoding="utf-8")

    assert "{language_rules}" not in prompt
    assert "Add punctuation and capitalization" in prompt


def test_text_stage_renders_only_active_language_rule() -> None:
    stage = TextLLMStage(
        prompt_text="{language}\n{language_rules}\n{text}",
        language_rules={"hi": "Hindi-only rule.", "ur": "Urdu-only rule."},
    )
    stage._system_prompt = stage._resolve_prompt()

    rendered = stage._render_prompt_template("नमस्ते दुनिया", {"source_lang": "hi"})

    assert rendered == "hi\nHindi-only rule.\nनमस्ते दुनिया"
    assert "Urdu-only rule." not in rendered


def test_remote_stage_uses_same_row_scoped_rendering() -> None:
    stage = RemoteTextLLMStage(
        prompt_text="{language}\n{language_rules}\n{text}",
        language_rules={"hi": "Hindi-only rule.", "ur": "Urdu-only rule."},
    )
    stage._system_prompt = stage._resolve_prompt()

    messages = stage._build_messages("नमस्ते दुनिया", {"source_lang": "hi"})

    assert messages == [{"role": "user", "content": "hi\nHindi-only rule.\nनमस्ते दुनिया"}]


@pytest.mark.parametrize("language", PNC_LANGUAGE_CODES)
def test_bundled_render_contains_only_active_language_guidance(language: str) -> None:
    stage = TextLLMStage(
        prompt_file=str(PROMPT_DIR / "pnc_prompt_indic.md"),
        language_rules=load_pnc_language_rules(),
    )
    stage._system_prompt = stage._resolve_prompt()

    rendered = stage._render_prompt_template("sample", {"source_lang": language})

    active_rule = load_pnc_language_rules()[language]
    assert active_rule in rendered
    for other_language in (
        "Assamese",
        "Bengali",
        "Bodo",
        "Dogri",
        "Gujarati",
        "Hindi",
        "Kannada",
        "Kashmiri",
        "Konkani",
        "Malayalam",
        "Maithili",
        "Manipuri",
        "Marathi",
        "Nepali",
        "Odia",
        "Punjabi",
        "Santali",
        "Sanskrit",
        "Sindhi",
        "Tamil",
        "Telugu",
        "Urdu",
    ):
        if other_language not in active_rule:
            assert other_language not in rendered


@pytest.mark.parametrize("task_data", [None, {}, {"source_lang": ""}, {"source_lang": "xx"}])
def test_language_rule_resolution_fails_closed(task_data: dict | None) -> None:
    stage = TextLLMStage(
        prompt_text="{language_rules}\n{text}",
        language_rules={"hi": "Hindi-only rule."},
    )
    stage._system_prompt = stage._resolve_prompt()

    with pytest.raises(ValueError, match=r"prompt requires|unsupported"):
        stage._render_prompt_template("नमस्ते दुनिया", task_data)


def test_prompt_without_language_rules_remains_backward_compatible() -> None:
    stage = TextLLMStage(prompt_text="{language}\n{text}")
    stage._system_prompt = stage._resolve_prompt()

    assert stage._render_prompt_template("hello", None) == "English\nhello"
