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

"""Load the per-row translated examples used by the Indic ITN prompt."""

from __future__ import annotations

import json
from pathlib import Path

ITN_LANGUAGE_CODES = (
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

DEFAULT_ITN_LANGUAGE_EXAMPLES_FILE = Path(__file__).parent / "prompts" / "itn_language_examples.json"


def load_itn_language_examples(path: str | Path = DEFAULT_ITN_LANGUAGE_EXAMPLES_FILE) -> dict[str, str]:
    """Return a deterministic, fail-closed mapping for the 22 target languages."""

    examples_path = Path(path)
    raw = json.loads(examples_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        message = f"ITN language examples must be a JSON object: {examples_path}"
        raise TypeError(message)

    found_codes = tuple(raw)
    if found_codes != ITN_LANGUAGE_CODES:
        message = f"Expected ITN language codes in order {ITN_LANGUAGE_CODES}, found {found_codes} in {examples_path}"
        raise ValueError(message)

    examples: dict[str, str] = {}
    for code, value in raw.items():
        if not isinstance(value, str) or not value.strip():
            message = f"ITN language examples {code!r} must be a non-empty string in {examples_path}"
            raise ValueError(message)
        examples[code] = value.strip()
    return examples
