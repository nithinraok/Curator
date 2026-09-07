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

"""LLM-based entity recovery stage using vLLM.

Given two transcriptions of the same audio — a *ground-truth* reference
(which keeps the correct casing/spelling of named entities) and a
*normalized* transcription (whose entities have been flattened, e.g.
``NVIDIA`` → ``nvidia``, ``New York`` → ``new york``) — this stage asks a
text LLM to rewrite the normalized transcription so its named entities
are restored to their ground-truth form, while every other part of the
normalized text is left untouched.

The ground-truth text is read from ``ground_truth_key`` and the
normalized text from ``normalized_key``; the recovered transcription is
written to ``output_text_key`` (default ``entity_recovered_text``). Both
input fields are preserved unchanged.

Engine ownership
----------------
Each pipeline stage runs as its own actor (its own process) under the
Ray Data / Xenna / Ray actor-pool executors, so this stage loads and
owns its own vLLM engine via ``_get_or_load_model`` — independent of the
other text-LLM stages, and therefore uses its own ``max_model_len`` /
sampling config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.audio.text_filtering.contextual_asr_extraction import _load_prompt_sections
from nemo_curator.stages.audio.text_filtering.text_llm_stage import _get_or_load_model
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

try:
    from vllm import SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "recover_entities_prompt.md"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_response(raw: str) -> str:
    """Strip reasoning blocks, code fences, and surrounding quotes from an LLM reply."""
    text = _THINK_BLOCK_RE.sub("", raw).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":  # noqa: PLR2004
        text = text[1:-1].strip()
    return text


@dataclass
class RecoverEntitiesStage(ProcessingStage[AudioTask, AudioTask]):
    """Recover named entities from a ground-truth transcription via an LLM.

    Reads a ground-truth transcription from ``ground_truth_key`` and a
    normalized transcription from ``normalized_key``, then prompts a
    text-only LLM to rewrite the normalized transcription so its named
    entities (proper nouns, acronyms, mixed-case brand names) use the
    ground-truth spelling/casing.  The rest of the normalized text is
    preserved.  The result is written to ``output_text_key``.

    Both input fields are preserved unchanged.  When ``_skipme`` is set,
    or when either transcription is empty, the normalized text is copied
    through to ``output_text_key`` untouched (no inference is run).

    Runs as its own actor/process and loads its own vLLM engine via the
    per-process cache in :mod:`text_llm_stage`.

    Args:
        model_id: HuggingFace model identifier for the text LLM.
        prompt_file: Path to the system+user prompt markdown file (with
            ``# SYSTEM_PROMPT`` and ``# USER_PROMPT_TEMPLATE`` sections).
            Falls back to the bundled default if not set.
        ground_truth_key: Manifest key holding the ground-truth transcription
            (source of the correct entity forms). Bound to ``{ground_truth}``.
        normalized_key: Manifest key holding the normalized transcription that
            entities are written into. Bound to ``{normalized}``.
        output_text_key: Output key for the recovered transcription.
        skip_me_key: Key used to check whether an entry is flagged.
        notes_key: Key holding the ``additional_notes`` dict written via
            :func:`set_note`.
        tensor_parallel_size: GPUs for tensor parallelism (None = auto).
        max_output_tokens: Maximum tokens to generate per sample.
        max_model_len: Maximum context length for this stage's vLLM engine.
        max_num_seqs: Maximum concurrent sequences in this stage's engine.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        kv_cache_dtype: KV-cache dtype for vLLM.
        num_workers_override: Explicit worker count for Xenna.
        batch_size: Number of samples per inference batch.
    """

    name: str = "RecoverEntities"
    model_id: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    prompt_file: str | None = None
    ground_truth_key: str = "ground_truth_text"
    normalized_key: str = "normalized_text"
    output_text_key: str = "entity_recovered_text"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    max_model_len: int = 4096
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.95
    kv_cache_dtype: str = "fp8"
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 64

    _llm: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _sampling_params: Any = field(default=None, init=False, repr=False)
    _system_prompt: str = field(default="", init=False, repr=False)
    _user_prompt_template: str = field(default="", init=False, repr=False)
    _n_processed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        tp = self.tensor_parallel_size
        if tp and tp > 0:
            self.resources = Resources(gpus=float(tp))

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def _resolve_prompts(self) -> tuple[str, str]:
        path = Path(self.prompt_file) if self.prompt_file else _DEFAULT_PROMPT_PATH
        if not path.exists():
            msg = f"RecoverEntities prompt file not found: {path}"
            raise FileNotFoundError(msg)
        return _load_prompt_sections(path)

    def _init_model(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vLLM is required for RecoverEntitiesStage. pip install vllm"
            raise ImportError(msg)

        self._system_prompt, self._user_prompt_template = self._resolve_prompts()

        from nemo_curator.utils.gpu_utils import get_gpu_count

        tp = self.tensor_parallel_size or get_gpu_count()
        self._llm, self._tokenizer = _get_or_load_model(
            model_id=self.model_id,
            tensor_parallel_size=tp,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_output_tokens,
        )
        logger.info(
            "%s: ready (system_prompt=%d chars, output_key=%s)",
            self.name,
            len(self._system_prompt),
            self.output_text_key,
        )

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        pass

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._llm is None:
            self._init_model()

    def teardown(self) -> None:
        if self._n_processed:
            logger.info("%s: processed %d samples", self.name, self._n_processed)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.ground_truth_key, self.normalized_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key]

    def _format_prompt(self, ground_truth: str, normalized: str) -> str:
        fmt: dict[str, str] = {}
        if "{ground_truth}" in self._user_prompt_template:
            fmt["ground_truth"] = ground_truth
        if "{normalized}" in self._user_prompt_template:
            fmt["normalized"] = normalized
        user_content = self._user_prompt_template.format(**fmt) if fmt else self._user_prompt_template
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self._llm is None:
            msg = "Model not initialised — setup() was not called"
            raise RuntimeError(msg)

        valid_indices: list[int] = []
        prompts: list[str] = []

        for i, task in enumerate(tasks):
            normalized = task.data.get(self.normalized_key, "")
            if not isinstance(normalized, str):
                normalized = ""
            ground_truth = task.data.get(self.ground_truth_key, "")
            if not isinstance(ground_truth, str):
                ground_truth = ""

            if task.data.get(self.skip_me_key, ""):
                task.data[self.output_text_key] = normalized
                set_note(task.data, self.name, "skipped (flagged)", self.notes_key)
                continue
            if not normalized.strip() or not ground_truth.strip():
                task.data[self.output_text_key] = normalized
                set_note(task.data, self.name, "skipped (empty)", self.notes_key)
                continue

            valid_indices.append(i)
            prompts.append(self._format_prompt(ground_truth.strip(), normalized.strip()))

        if prompts:
            outputs = self._llm.generate(prompts, sampling_params=self._sampling_params, use_tqdm=False)

            for seq_idx, task_idx in enumerate(valid_indices):
                task = tasks[task_idx]
                normalized = task.data[self.normalized_key]
                result_text = _clean_response(outputs[seq_idx].outputs[0].text)
                # Guard against empty / degenerate generations: fall back to the
                # normalized text so we never drop content.
                if not result_text:
                    task.data[self.output_text_key] = normalized
                    set_note(task.data, self.name, "fallback (empty output)", self.notes_key)
                else:
                    task.data[self.output_text_key] = result_text
                    note = "recovered" if result_text != normalized else "unchanged"
                    set_note(task.data, self.name, note, self.notes_key)
                self._n_processed += 1

        logger.debug("%s: batch of %d tasks (%d inferred)", self.name, len(tasks), len(prompts))
        return tasks
