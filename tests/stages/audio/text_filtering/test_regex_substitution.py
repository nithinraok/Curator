# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import pytest
import yaml

from nemo_curator.stages.audio.text_filtering.regex_substitution import RegexSubstitutionStage
from nemo_curator.tasks import AudioTask


def _write_rules(tmp_path: Path, rules: list[dict]) -> str:
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.dump(rules), encoding="utf-8")
    return str(p)


def test_applies_substitution(tmp_path: Path) -> None:
    rules_path = _write_rules(tmp_path, [{"pattern": "\u2019", "repl": "'"}])
    stage = RegexSubstitutionStage(regex_params_yaml=rules_path, text_key="cleaned_text", skip_me_key="skip_me")
    stage.setup()
    task = AudioTask(data={"cleaned_text": "it\u2019s fine", "skip_me": ""})
    result = stage.process(task)
    assert "'" in result.data["cleaned_text"]
    assert result.data["skip_me"] == ""


def test_empty_text_after_rules_sets_skip_me(tmp_path: Path) -> None:
    rules_path = _write_rules(tmp_path, [{"pattern": r"\w+", "repl": ""}])
    stage = RegexSubstitutionStage(regex_params_yaml=rules_path, text_key="cleaned_text", skip_me_key="skip_me")
    stage.setup()
    task = AudioTask(data={"cleaned_text": "hello", "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == "Empty after regex cleaning"


def test_whitespace_only_sets_skip_me(tmp_path: Path) -> None:
    rules_path = _write_rules(tmp_path, [{"pattern": r"\S+", "repl": ""}])
    stage = RegexSubstitutionStage(regex_params_yaml=rules_path, text_key="cleaned_text", skip_me_key="skip_me")
    stage.setup()
    task = AudioTask(data={"cleaned_text": "hello world", "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == "Empty after regex cleaning"


def test_already_empty_prediction_does_not_set_skip_me(tmp_path: Path) -> None:
    # A prediction that was ALREADY empty (or whitespace-only) before cleaning must NOT be
    # flagged "Empty after regex cleaning" — the regex removed nothing, so skip_me stays "".
    rules_path = _write_rules(tmp_path, [{"pattern": r"\w+", "repl": ""}])
    stage = RegexSubstitutionStage(regex_params_yaml=rules_path, text_key="cleaned_text", skip_me_key="skip_me")
    stage.setup()
    for empty_pred in ("", "   "):
        task = AudioTask(data={"cleaned_text": empty_pred, "skip_me": ""})
        result = stage.process(task)
        assert result.data["cleaned_text"] == ""
        assert result.data["skip_me"] == ""


def test_non_empty_text_preserves_skip_me_empty(tmp_path: Path) -> None:
    rules_path = _write_rules(tmp_path, [{"pattern": r"bad", "repl": "good"}])
    stage = RegexSubstitutionStage(regex_params_yaml=rules_path, text_key="cleaned_text", skip_me_key="skip_me")
    stage.setup()
    task = AudioTask(data={"cleaned_text": "bad word", "skip_me": ""})
    result = stage.process(task)
    assert result.data["cleaned_text"] == "good word"
    assert result.data["skip_me"] == ""


def test_strips_extra_whitespace(tmp_path: Path) -> None:
    rules_path = _write_rules(tmp_path, [])
    stage = RegexSubstitutionStage(regex_params_yaml=rules_path, text_key="cleaned_text", skip_me_key="skip_me")
    stage.setup()
    task = AudioTask(data={"cleaned_text": "hello   world", "skip_me": ""})
    result = stage.process(task)
    assert result.data["cleaned_text"] == "hello world"


def test_multiple_rules_applied_in_order(tmp_path: Path) -> None:
    rules_path = _write_rules(
        tmp_path,
        [
            {"pattern": "\u2014", "repl": "-"},
            {"pattern": r"\s+", "repl": " "},
        ],
    )
    stage = RegexSubstitutionStage(regex_params_yaml=rules_path, text_key="cleaned_text", skip_me_key="skip_me")
    stage.setup()
    task = AudioTask(data={"cleaned_text": "word\u2014word", "skip_me": ""})
    result = stage.process(task)
    assert result.data["cleaned_text"] == "word-word"


def test_setup_called_lazily_when_skipped(tmp_path: Path) -> None:
    rules_path = _write_rules(tmp_path, [{"pattern": "\u2019", "repl": "'"}])
    stage = RegexSubstitutionStage(regex_params_yaml=rules_path, text_key="cleaned_text", skip_me_key="skip_me")
    task = AudioTask(data={"cleaned_text": "it\u2019s fine", "skip_me": ""})
    result = stage.process(task)
    assert result.data["cleaned_text"] == "it's fine"


def test_non_string_text_returns_task_unchanged(tmp_path: Path) -> None:
    rules_path = _write_rules(tmp_path, [{"pattern": r"\w+", "repl": ""}])
    stage = RegexSubstitutionStage(regex_params_yaml=rules_path, text_key="cleaned_text", skip_me_key="skip_me")
    stage.setup()
    task = AudioTask(data={"cleaned_text": None, "skip_me": ""})
    result = stage.process(task)
    assert result.data["cleaned_text"] is None
    assert result.data["skip_me"] == ""


def test_requires_regex_params_yaml() -> None:
    with pytest.raises(ValueError, match="regex_params_yaml is required"):
        RegexSubstitutionStage(regex_params_yaml="")


def test_preserves_existing_skip_me_on_empty_result(tmp_path: Path) -> None:
    rules_path = _write_rules(tmp_path, [{"pattern": r"\w+", "repl": ""}])
    stage = RegexSubstitutionStage(regex_params_yaml=rules_path, text_key="cleaned_text", skip_me_key="skip_me")
    stage.setup()
    task = AudioTask(data={"cleaned_text": "hello", "skip_me": "Hallucination"})
    result = stage.process(task)
    assert result.data["skip_me"] == "Hallucination"


_COMMON_YAML = (
    Path(__file__).resolve().parents[4] / "tutorials" / "audio" / "granary_v2_postprocessing" / "common.yaml"
)


@pytest.mark.parametrize(
    "text",
    [
        "cũ kỹ nhưng vẫn dùng được",  # "old" — has ũ
        "nghĩ về chuyện cũ",  # "think about the old thing" — has ĩ and ũ
        "một bác sĩ giỏi",  # "a good doctor" — has ĩ
    ],
)
def test_common_yaml_preserves_vietnamese_plain_tilde_vowels(text: str) -> None:
    """Regression test: U+0128/0129 (Ĩĩ) and U+0168/0169 (Ũũ) must survive the production
    character whitelist. These were previously missing from the whitelist's Vietnamese
    coverage, so the regex silently stripped them (e.g. "cũ" -> "c "), even though the ASR
    prediction was correct — and the LLM post-processing stages only partially recovered
    the loss downstream.
    """
    stage = RegexSubstitutionStage(regex_params_yaml=str(_COMMON_YAML), text_key="cleaned_text")
    stage.setup()
    task = AudioTask(data={"cleaned_text": text, "_skipme": ""})
    result = stage.process(task)
    assert "ĩ" in result.data["cleaned_text"] or "ĩ" not in text
    assert "ũ" in result.data["cleaned_text"] or "ũ" not in text
    # No word should have been reduced to nothing by the whitelist.
    assert len(result.data["cleaned_text"].split()) == len(text.split())
