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

"""Load the compact per-row language guidance used by the PnC prompt."""

from __future__ import annotations

import json
from pathlib import Path

PNC_LANGUAGE_CODES = (
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

DEFAULT_PNC_LANGUAGE_RULES_FILE = Path(__file__).parent / "prompts" / "pnc_language_rules.json"


def load_pnc_language_rules(path: str | Path = DEFAULT_PNC_LANGUAGE_RULES_FILE) -> dict[str, str]:
    """Return a deterministic, fail-closed mapping for the 22 target languages."""

    rules_path = Path(path)
    raw = json.loads(rules_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        message = f"PnC language rules must be a JSON object: {rules_path}"
        raise TypeError(message)

    found_codes = tuple(raw)
    if found_codes != PNC_LANGUAGE_CODES:
        message = f"Expected PnC language codes in order {PNC_LANGUAGE_CODES}, found {found_codes} in {rules_path}"
        raise ValueError(message)

    rules: dict[str, str] = {}
    for code, value in raw.items():
        if not isinstance(value, str) or not value.strip():
            message = f"PnC language rule {code!r} must be a non-empty string in {rules_path}"
            raise ValueError(message)
        rules[code] = value.strip()
    return rules
