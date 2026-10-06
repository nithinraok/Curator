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

"""Batched language identification with AI4Bharat IndicLID-FTN v1.0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from nemo_curator.stages.audio.text_filtering.fasttext_language_identification import (
    _FastTextLanguageIdentificationStage,
)

# Literal label inventory of the released IndicLID-FTN v1.0 checkpoint. The
# historical script spellings (Mlym, Meti, Olch, Orya) intentionally match the
# model rather than modern ISO 15924 spellings.
INDICLID_LABEL_TO_LANGUAGE: dict[str, str] = {
    "asm_Beng": "as",
    "ben_Beng": "bn",
    "brx_Deva": "brx",
    "doi_Deva": "doi",
    # Production manifests/verifier use ``en``. The exploratory Granary report
    # rendered this checkpoint label as ``eng``; no evaluated Indic prediction
    # changes, and canonicalizing here preserves the LLM-stage row contract.
    "eng_Latn": "en",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "kan_Knda": "kn",
    "kas_Arab": "ks",
    "kas_Deva": "ks",
    "kok_Deva": "kok",
    "mai_Deva": "mai",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "mni",
    "mni_Meti": "mni",
    "nep_Deva": "ne",
    "ori_Orya": "or",
    "other": "other",
    "pan_Guru": "pa",
    "san_Deva": "sa",
    "sat_Olch": "sat",
    "snd_Arab": "sd",
    "tam_Tamil": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}


@dataclass
class IndicLIDLanguageIdentificationStage(_FastTextLanguageIdentificationStage):
    """Run top-1 IndicLID-FTN inference with local-evaluation behavior.

    Only carriage returns and line feeds are replaced with spaces. The output
    receives the normalized manifest language code. Kashmiri and Manipuri's
    two script labels map to the same language code.
    """

    name: str = "LanguageID"

    backend_name: ClassVar[str] = "indiclid"
    label_to_language: ClassVar[dict[str, str]] = INDICLID_LABEL_TO_LANGUAGE

    def _validate_model_labels(self, labels: frozenset[str]) -> None:
        expected_labels = frozenset(self.label_to_language)
        if labels != expected_labels:
            msg = (
                "IndicLID-FTN label inventory mismatch: "
                f"missing={sorted(expected_labels - labels)}, extra={sorted(labels - expected_labels)}"
            )
            raise ValueError(msg)

    def _sanitize_text(self, text: str) -> str:
        return text.replace("\r", " ").replace("\n", " ")

    def _map_label_to_language(self, raw_label: str) -> str:
        return self.label_to_language[raw_label]

    def _validate_source_language(self, language: str) -> None:
        if language not in self.label_to_language.values():
            msg = f"IndicLID-FTN has no label mapping for source_lang={language!r}"
            raise ValueError(msg)
