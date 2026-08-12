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

from nemo_curator.stages.audio.text_filtering.whisper_hallucination import WhisperHallucinationStage
from nemo_curator.tasks import AudioTask


def _make_stage(tmp_path: Path, phrases: list[str], **kwargs) -> WhisperHallucinationStage:
    p = tmp_path / "phrases.txt"
    p.write_text("\n".join(phrases), encoding="utf-8")
    stage = WhisperHallucinationStage(common_hall_file=str(p), **kwargs)
    stage.setup()
    return stage


_TEXT_KEY = "pred_text"
_SKIP_KEY = "_skipme"


def test_clean_text_passes(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["lorem ipsum"])
    task = AudioTask(data={_TEXT_KEY: "the cat sat on the mat today", _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_repeated_ngrams_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_long_word_absolute_threshold_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    long_word = "a" * 30
    task = AudioTask(data={_TEXT_KEY: f"the {long_word} here", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_long_word_relative_threshold_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "cat verylongwordindeed", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_frequent_phrase_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you"])
    task = AudioTask(data={_TEXT_KEY: "Thank you", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_frequent_phrase_strips_punctuation(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you"])
    task = AudioTask(data={_TEXT_KEY: "Thank you.", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_frequent_phrase_strips_trailing_comma(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you"])
    task = AudioTask(data={_TEXT_KEY: "Thank you,", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_setup_called_lazily_when_skipped(tmp_path: Path) -> None:
    p = tmp_path / "phrases.txt"
    p.write_text("Thank you\n", encoding="utf-8")
    stage = WhisperHallucinationStage(common_hall_file=str(p))
    task = AudioTask(data={_TEXT_KEY: "Thank you", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_non_string_text_returns_task_unchanged(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: None, _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_preserves_existing_skip_me_reason(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: "Wrong language"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == "Wrong language"


def test_empty_words_not_flagged_by_ngram(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    assert stage._repeated_ngrams([]) is False


def test_empty_words_not_flagged_by_long_word(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    assert stage._long_word([]) is False


def test_phrases_file_loads_lines_as_is(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you", "Amen", "Yeah"])
    assert "Thank you" in stage._phrases
    assert "Amen" in stage._phrases
    assert "Yeah" in stage._phrases


def test_requires_common_hall_file() -> None:
    with pytest.raises(ValueError, match="common_hall_file is required"):
        WhisperHallucinationStage(common_hall_file="")


def test_agglutinative_lang_uses_higher_threshold(tmp_path: Path) -> None:
    """A 34-char word in Finnish should NOT be flagged (threshold=35 for agglutinative)."""
    stage = _make_stage(tmp_path, [])
    word_34 = "a" * 34
    task = AudioTask(data={_TEXT_KEY: f"the {word_34} here", _SKIP_KEY: "", "language": "fi"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_agglutinative_lang_flags_above_threshold(tmp_path: Path) -> None:
    """A 36-char word in Finnish SHOULD be flagged (above threshold=35)."""
    stage = _make_stage(tmp_path, [])
    word_36 = "a" * 36
    task = AudioTask(data={_TEXT_KEY: f"the {word_36} here", _SKIP_KEY: "", "language": "fi"})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_agglutinative_lang_skips_relative_check(tmp_path: Path) -> None:
    """A long word in Hungarian shouldn't trigger relative threshold (skip_relative=True)."""
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "a természetvédelmi cat", _SKIP_KEY: "", "language": "hu"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_non_agglutinative_lang_uses_default_threshold(tmp_path: Path) -> None:
    """A 26-char word in English SHOULD be flagged (default threshold=25)."""
    stage = _make_stage(tmp_path, [])
    word_26 = "a" * 26
    task = AudioTask(data={_TEXT_KEY: f"the {word_26} here", _SKIP_KEY: "", "language": "en"})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


# --- scriptio continua (space-less) languages -------------------------------------------------
# These are written without spaces, so text.split() returns the whole utterance as ONE token.
# The word-level checks must not run: _long_word would flag every utterance over the threshold.

_JA_LONG = "なんかおばあちゃんのレシピみたいなあのどんな思い出とか食べ物とかでどんな思い出とかあったりしますか"


def test_scriptio_continua_long_utterance_not_flagged(tmp_path: Path) -> None:
    """A 50-char Japanese sentence is one whitespace token — must NOT trip _long_word."""
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: _JA_LONG, _SKIP_KEY: "", "language": "ja"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


@pytest.mark.parametrize("lang", ["ja", "zh", "th", "zh-CN", "yue", "Japanese"])
def test_scriptio_continua_langs_skip_word_checks(tmp_path: Path, lang: str) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "字" * 60, _SKIP_KEY: "", "language": lang})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_same_text_without_language_is_still_flagged(tmp_path: Path) -> None:
    """Control: the skip is driven by the language field, not by the text itself."""
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: _JA_LONG, _SKIP_KEY: "", "language": "en"})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_scriptio_continua_still_flags_phrase_match(tmp_path: Path) -> None:
    """Character-level checks stay active for space-less languages."""
    stage = _make_stage(tmp_path, ["ご視聴ありがとうございました"])
    task = AudioTask(data={_TEXT_KEY: "ご視聴ありがとうございました", _SKIP_KEY: "", "language": "ja"})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_scriptio_continua_still_flags_high_char_rate(tmp_path: Path) -> None:
    """High char rate is character-based and must keep firing for space-less languages."""
    stage = _make_stage(tmp_path, [], max_char_rate=40.0)
    task = AudioTask(data={_TEXT_KEY: "字" * 100, _SKIP_KEY: "", "language": "ja", "duration": 1.0})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]
