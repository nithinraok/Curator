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
        "mình cũng biết mà",  # vi: ũ U+0169
        "nghĩ về chuyện cũ",  # vi: ĩ U+0129 and ũ
        "một bác sĩ giỏi",  # vi: ĩ
        "NOËL",  # fr: Ë U+00CB
        "AOÛT",  # fr: Û U+00DB
        "MAÏS",  # fr: Ï U+00CF
        "όλους τους ανθρώπους",  # el: final sigma ς U+03C2
        "Ίσως Ύστερα Ώρα",  # el: Ί U+038A, Ύ U+038E, Ώ U+038F
        "προϋπόθεση ταΐζω",  # el: ϋ U+03CB, ΐ U+0390
        "ΠΡΟΫΠΟΘΕΣΗ ΤΑΪΖΩ",  # el: Ϋ U+03AB, Ϊ U+03AA
        "naţiune Ţara",  # ro: legacy cedilla ţ U+0163, Ţ U+0162
        "Α΄ Παγκόσμιος Πόλεμος",  # el: numeral sign/keraia ΄ U+0384
        "тя ѝ каза",  # bg: ѝ U+045D
        "Ѝ и ѐ",  # bg: Ѝ U+040D, ѐ U+0450
        "ĳsje",  # nl: ĳ U+0133
        "人々は時々",  # ja: 々 U+3005
        "二〇二四年",  # zh/ja: 〇 U+3007
    ],
)
def test_production_whitelist_preserves_covered_language_letters(text: str) -> None:
    """Letters absent from the whitelist are silently replaced by a space.

    Ĩĩ/Ũũ and the uppercase Ë/Ï/Û were missing, so correct predictions lost them
    mid-word: "cũng" became "c ng" and "NOËL" became "NO L". The same gap hit the
    Greek final sigma ς ("τους" -> "του "), Romanian cedilla ţ, Bulgarian ѝ and the
    CJK iteration mark 々 ("人々" -> "人 "). Every one of these languages is in the
    rule file's declared coverage, so a covered letter must survive intact.
    """
    stage = RegexSubstitutionStage(
        regex_params_yaml=str(_COMMON_YAML),
        text_key="pred_text",
        output_text_key="cleaned_text",
    )
    stage.setup()
    task = AudioTask(data={"pred_text": text})

    stage.process(task)

    assert task.data["cleaned_text"] == text
