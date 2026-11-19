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

from nemo_curator.stages.audio.text_filtering.abbreviation_concat import (
    AbbreviationConcatStage,
    concat_abbreviations,
)
from nemo_curator.tasks import AudioTask


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ("the A P I uses G P U acceleration", "en", "the API uses GPU acceleration"),
        ("the U K's policy", "en", "the UK's policy"),
        ("A P Xs", "en", "APXs"),
        ("A B Is", "en", "AB Is"),
        ("A P As", "en", "AP As"),
        ("A Is", "en", "A Is"),
        ("a A P I", "en", "a API"),
        ("A P I a", "en", "API a"),
        ("D N a and R N a", "en", "DN a and RN a"),
        ("I I and A A", "en", "I I and A A"),
        ("x I and A x", "en", "x I and A x"),
        ("I'm A B", "en", "I'm AB"),
        ("i'm A B", "en", "i'm AB"),
        ("I\u2018m A B", "en", "I\u2018m AB"),
        ("'A B C", "en", "'ABC"),
        ("\u2018A B C\u2019", "en", "\u2018ABC\u2019"),
        ("\u2019A B C\u2019", "en", "\u2019A BC\u2019"),
        ("x'A B", "en", "x'AB"),
        ("x\u2018A B", "en", "x\u2018AB"),
        ("x\u2019A B", "en", "x\u2019A B"),
        ("A B i'm", "en", "AB i'm"),
        ("A B i\u2019m", "en", "AB i\u2019m"),
        ("A B I\u2019s", "en", "ABI\u2019s"),
        ("a A Is", "en", "a A Is"),
        ("a a B", "en", "a a B"),
        ("B a a", "en", "B a a"),
        ("A B bs", "en", "AB bs"),
        ("a bs", "en", "a bs"),
        ("А Б В", "ru", "АБВ"),  # noqa: RUF001
        ("Α Β Γ", "el", "ΑΒΓ"),  # noqa: RUF001
        ("a cat sat nearby", "en", "a cat sat nearby"),
        # Lowercase single letters are function words, never abbreviations.
        ("il y a deux ans", "fr", "il y a deux ans"),
        ("il n'y a pas", "fr", "il n'y a pas"),
        ("il va à l'école", "fr", "il va à l'école"),
        ("il y a d'autres", "fr", "il y a d'autres"),
        ("la S N C F à l'heure", "fr", "la SNCF à l'heure"),
        ("de A à Z", "fr", "de A à Z"),
        ("l'O N U", "fr", "l'ONU"),
        ("o azoto é o elemento", "pt", "o azoto é o elemento"),
        ("questa è l'unica", "it", "questa è l'unica"),
        ("A y B", "es", "A y B"),
        ("i w stawie", "pl", "i w stawie"),
        ("a v rámci", "cs", "a v rámci"),
        ("mi s a părut", "ro", "mi s a părut"),
        ("и в том числе", "ru", "и в том числе"),
        ("і в червоношкірих", "uk", "і в червоношкірих"),  # noqa: RUF001
        ("one p m in Chicago", "en", "one p m in Chicago"),
    ],
)
def test_concat_abbreviations(text: str, language: str, expected: str) -> None:
    result, _ = concat_abbreviations(text, language=language)

    assert result == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A P I and G P U", ["API", "GPU"]),
        ("a A P I", ["API"]),
        ("A P Xs", ["APX"]),
        ("A B Is", ["AB I"]),
        ("A B bs", ["AB"]),
        ("\u2019A B C\u2019", ["BC"]),
        ("x'A B", ["AB"]),
    ],
)
def test_reports_each_changed_abbreviation(text: str, expected: list[str]) -> None:
    _, found = concat_abbreviations(text)

    assert found == expected


def test_stage_records_changed_abbreviations() -> None:
    task = AudioTask(data={"text": "il y a la S N C F", "source_lang": "fr"})

    AbbreviationConcatStage().process(task)

    assert task.data["abbreviated_text"] == "il y a la SNCF"
    assert task.data["additional_notes"]["AbbreviationConcat"] == "applied (S N C F -> SNCF)"


def test_stage_leaves_lowercase_function_words_alone() -> None:
    task = AudioTask(data={"text": "il y a deux ans", "source_lang": "fr"})

    AbbreviationConcatStage().process(task)

    assert task.data["abbreviated_text"] == "il y a deux ans"
    assert "additional_notes" not in task.data
