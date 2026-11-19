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

import re
from dataclasses import dataclass, field
from typing import Any

import yaml
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class RegexSubstitutionStage(ProcessingStage[AudioTask, AudioTask]):
    """Apply a sequence of regex substitutions to a text field in each AudioTask.

    Rules are loaded from a YAML file containing a list of dicts with
    ``pattern`` and ``repl`` keys (and an optional ``count`` key). A rule may also define
    ``langs`` (apply only to those languages) or ``exclude_langs`` (apply to every language
    except those), each a list of codes matched against ``language_key``; a region suffix such
    as ``el-GR`` matches ``el``. A row without a language gets every rule that has no ``langs``.

    Reads from ``text_key`` (default ``"pred_text"``), applies all rules,
    and writes the result to ``output_text_key`` (default ``"cleaned_text"``).
    The original ``text_key`` field is preserved unchanged.

    After all substitutions, ``_skipme`` is set to ``"Empty after regex cleaning"``
    ONLY when the regex actually removed content: the prediction was non-empty before
    cleaning, became empty after, and ``_skipme`` was still empty. A prediction that was
    already empty before cleaning is left untouched (``_skipme`` is NOT set) — nothing was
    cleaned away, so it is not a regex-cleaning skip.
    """

    regex_params_yaml: str = ""
    text_key: str = "pred_text"
    output_text_key: str = "cleaned_text"
    skip_me_key: str = "_skipme"
    language_key: str = "source_lang"
    name: str = "RegexSubstitution"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _rules: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _setup_called: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.regex_params_yaml:
            msg = "regex_params_yaml is required for RegexSubstitutionStage"
            raise ValueError(msg)

    def setup(self, _worker_metadata: object | None = None) -> None:
        with open(self.regex_params_yaml, encoding="utf-8") as f:
            self._rules = yaml.safe_load(f)
        for index, rule in enumerate(self._rules):
            self._validate_language_filter(index, rule)
        self._setup_called = True
        logger.info(f"RegexSubstitutionStage: loaded {len(self._rules)} rules from {self.regex_params_yaml}")

    @staticmethod
    def _validate_language_filter(index: int, rule: dict[str, Any]) -> None:
        if "langs" in rule and "exclude_langs" in rule:
            msg = f"Regex rule {index} must not define both langs and exclude_langs"
            raise ValueError(msg)
        for key in ("langs", "exclude_langs"):
            langs = rule.get(key)
            if key in rule and not (
                isinstance(langs, list) and langs and all(isinstance(x, str) and x for x in langs)
            ):
                msg = f"Regex rule {index} {key} must be a non-empty list of language codes"
                raise TypeError(msg)

    @staticmethod
    def _applies(rule: dict[str, Any], language: str | None) -> bool:
        if "langs" in rule:
            return language in {x.lower() for x in rule["langs"]}
        if "exclude_langs" in rule:
            return language not in {x.lower() for x in rule["exclude_langs"]}
        return True

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key, self.skip_me_key]

    def _process_single(self, task: AudioTask) -> AudioTask:
        if task.data.get(self.skip_me_key, ""):
            task.data.setdefault(self.output_text_key, "")
            return task
        text = task.data[self.text_key]
        if not isinstance(text, str):
            return task
        had_text = bool(text.strip())
        lang = task.data.get(self.language_key)
        lang = re.split(r"[-_]", lang.strip(), maxsplit=1)[0].lower() or None if isinstance(lang, str) else None
        text = " " + text + " "
        for rule in self._rules:
            if not self._applies(rule, lang):
                continue
            text = re.sub(rule["pattern"], rule["repl"], text, count=rule.get("count", 0))
        text = re.sub(r"\s+", " ", text).strip()
        task.data[self.output_text_key] = text
        if not text and had_text and not task.data.get(self.skip_me_key, ""):
            task.data[self.skip_me_key] = "Empty after regex cleaning"
        return task

    def process(self, task: AudioTask) -> AudioTask:
        if not self._setup_called:
            logger.warning(
                f"RegexSubstitutionStage ({self.name}): setup() was not called before process(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        return self._process_single(task)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not self._setup_called:
            logger.warning(
                f"RegexSubstitutionStage ({self.name}): setup() was not called before process_batch(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        return [self._process_single(task) for task in tasks]
