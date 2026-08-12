# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nemo_curator.stages.audio.text_filtering.select_best_prediction import SelectBestPredictionStage
from nemo_curator.tasks import AudioTask

_PRIMARY = "primary_model_prediction"
_FALLBACK = "fallback_model_prediction"
_SKIP = "_skipme"
_BEST = "best_prediction"
_METRIC = "omni_asr_agreement_metric"
_RATE = "omni_asr_agreement_wer"

# Real ja pair from a production manifest: the models differ by one dropped character
# ("おばあちゃん" vs "ばあちゃん") and a trailing "。". Word-level WER sees two single-token
# texts that are not byte-identical and returns 100; CER returns ~2.
_JA_PRIMARY = "なんかおばあちゃんのレシピみたいなあのどんな思い出とか食べ物とかでどんな思い出とかあったりしますか"
_JA_FALLBACK = "なんかばあちゃんのレシピみたいなあのどんな思い出とか食べ物とかでどんな思い出とかあったりしますか。"


def _task(primary: str, fallback: str, lang: str) -> AudioTask:
    return AudioTask(
        data={
            _PRIMARY: primary,
            _FALLBACK: fallback,
            _SKIP: "Hallucination:WhisperHallucination_asr",
            "source_lang": lang,
            "additional_notes": {},
        }
    )


def test_cjk_near_identical_predictions_are_recovered() -> None:
    """The bug this fixes: word-level WER is 100 here, so the sample was silently dropped."""
    stage = SelectBestPredictionStage()
    result = stage.process(_task(_JA_PRIMARY, _JA_FALLBACK, "ja"))

    assert result.data[_SKIP] == ""
    assert result.data[_BEST] == _JA_PRIMARY
    assert result.data[_METRIC] == "cer"
    assert result.data[_RATE] < 20.0  # noqa: PLR2004


@pytest.mark.parametrize("lang", ["ja", "zh", "th", "zh-TW", "yue"])
def test_space_less_languages_use_cer(lang: str) -> None:
    stage = SelectBestPredictionStage()
    result = stage.process(_task("这是一个测试句子用来检查协议", "这是一个测试句子用来检查协义", lang))
    assert result.data[_METRIC] == "cer"
    assert result.data[_SKIP] == ""


@pytest.mark.parametrize("lang", ["en", "de", "es", "ko", "vi"])
def test_space_separated_languages_still_use_wer(lang: str) -> None:
    """Regression guard: behaviour for space-separated languages must not change."""
    stage = SelectBestPredictionStage()
    result = stage.process(_task("the cat sat on the mat", "the cat sat on the mat", lang))
    assert result.data[_METRIC] == "wer"
    assert result.data[_SKIP] == ""


def test_cjk_genuine_disagreement_stays_flagged() -> None:
    """CER is not a blanket amnesty — unrelated predictions must remain flagged."""
    stage = SelectBestPredictionStage()
    result = stage.process(_task("あなたのおすすめの映画は何ですか", "今日はとても良い天気ですね", "ja"))

    assert result.data[_SKIP].startswith("Hallucination")
    assert result.data[_METRIC] == "cer"
    assert result.data[_RATE] > 20.0  # noqa: PLR2004


def test_english_genuine_disagreement_stays_flagged() -> None:
    stage = SelectBestPredictionStage()
    result = stage.process(_task("the cat sat on the mat", "please subscribe to my channel", "en"))
    assert result.data[_SKIP].startswith("Hallucination")


def test_missing_language_falls_back_to_wer() -> None:
    """No source_lang field must not raise and must keep the previous WER behaviour."""
    stage = SelectBestPredictionStage()
    task = AudioTask(
        data={
            _PRIMARY: "hello world",
            _FALLBACK: "hello world",
            _SKIP: "Hallucination:WhisperHallucination_asr",
            "additional_notes": {},
        }
    )
    result = stage.process(task)
    assert result.data[_METRIC] == "wer"
    assert result.data[_SKIP] == ""
