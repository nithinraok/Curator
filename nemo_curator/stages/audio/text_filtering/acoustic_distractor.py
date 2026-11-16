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

"""Acoustic distractor generation stage for contextual ASR (CPU-only).

Appends phonetically-similar words/phrases to the ``distractor_terms``
list produced by :class:`ContextualASRExtractionStage`.  Semantic
distractors from the LLM teach the model not to copy hints blindly;
acoustic distractors additionally teach the model to disambiguate
phonetically-confusable words.

For each entity in ``fine_context_terms``:

1. G2P the full phrase via a configured permissive backend into a phone
   token list.
2. Compute Normalized Phonetic Distance (NPD)
   ``editdistance(query, candidate) / len(query)`` against every entry in
   a precomputed phoneme vocabulary loaded at ``setup()``.
3. Filter by ``min_npd < NPD < max_npd`` to drop both trivially-identical
   matches and too-distant ones.
4. Exclude vocabulary entries already present in ``fine_context_terms``
   or the existing (semantic) ``distractor_terms``.
5. Take the top ``per_entity_top_k`` candidates by ascending NPD.

Candidates collected across all entities of a sample are merged,
deduplicated, sorted by NPD, capped at ``max_acoustic_distractors``, and
appended to ``distractor_terms``.  The combined list is then capped at
``max_total_distractors``.

The precomputed phoneme vocabulary is produced offline by
``scripts/build_phoneme_vocab.py``.  Build it once per target language.
``phoneme_vocab_path`` may point either at a single JSON file (one language,
used for every sample) or at a **directory** of ``phoneme_vocab_<lang>.json``
files — in directory mode every file is loaded and the per-sample
``source_lang`` selects the matching vocab, so a single stage instance can
serve a multi-language job.  New vocab files include metadata describing the
backend/language used to build them; legacy plain ``{word: [phones]}`` files
are still accepted and use the stage-level backend settings.

This stage is CPU-only — no GPU or LLM is required at runtime.

Language handling
-----------------
The stage needs a language to G2P entities.  In order
of precedence:

1. Vocab metadata, when present.
2. ``language`` (if set on the stage) — used for all samples.
3. ``source_lang`` from the manifest (display name like ``"English"``
   or ISO-639-1 code like ``"en"``).
4. ``default_source_lang`` — used when neither of the above resolves.

When a sample's language cannot be mapped to a supported G2P language,
the stage records ``unsupported_language`` in the additional_notes and
leaves the existing ``distractor_terms`` untouched.

Dependencies
------------
- ``editdistance`` (already a transitive project dependency)
- Optional G2P backends: Montreal Forced Aligner CLI + CC-BY G2P models,
  ``phonikud``, ``g2p-en``, ``pypinyin``, NRC-ILT ``g2p``, ``epitran``, or
  ``segments``.  The built-in ``rules`` backend provides approximate fallback
  coverage for all target languages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.audio.text_filtering.g2p_backend import (
    BaseG2PBackend,
    G2PBackendConfig,
    G2PError,
    build_g2p_config,
    build_g2p_config_from_metadata,
    make_g2p_backend,
    normalize_g2p_language,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

try:
    import editdistance as _editdistance

    EDITDISTANCE_AVAILABLE = True
except ImportError:
    EDITDISTANCE_AVAILABLE = False
    _editdistance = None  # type: ignore[assignment]


@dataclass
class _VocabBundle:
    items: list[tuple[str, list[str]]]
    g2p_config: G2PBackendConfig | None
    metadata: dict[str, Any]


def _npd(query: list[str], candidate: list[str]) -> float:
    if not query:
        return 1.0
    if _editdistance is None:
        return 1.0
    return _editdistance.eval(query, candidate) / len(query)


def _vocab_search(  # noqa: PLR0913
    query_phonemes: list[str],
    vocab_items: list[tuple[str, list[str]]],
    *,
    min_npd: float,
    max_npd: float,
    excluded_words: set[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """Brute-force nearest-neighbor search in the precomputed phoneme vocab.

    Returns up to ``top_k`` ``(word, npd)`` pairs with the smallest NPD
    satisfying ``min_npd < npd < max_npd``.  Words in ``excluded_words``
    (case-insensitive match on the vocab key) are skipped.
    """
    if not query_phonemes or top_k <= 0:
        return []
    q_len = len(query_phonemes)
    len_lo = max(1, int(q_len * (1.0 - max_npd)))
    len_hi = max(1, int(q_len * (1.0 + max_npd)) + 1)

    hits: list[tuple[str, float]] = []
    for word, phonemes in vocab_items:
        if word.lower() in excluded_words:
            continue
        if not (len_lo <= len(phonemes) <= len_hi):
            continue
        d = _npd(query_phonemes, phonemes)
        if min_npd < d < max_npd:
            hits.append((word, d))

    hits.sort(key=lambda kv: kv[1])
    return hits[:top_k]


@dataclass
class AcousticDistractorStage(ProcessingStage[AudioTask, AudioTask]):
    """Append phonetically-similar distractors to ``context_asr.distractor_terms``.

    CPU-only stage.  Loads a precomputed phoneme vocabulary at
    ``setup()`` and, for each task, G2Ps every entity in
    ``fine_context_terms``, searches the vocab for phonetically similar
    words by Normalized Phonetic Distance (NPD), and appends the top
    candidates to ``distractor_terms`` (capped at
    ``max_total_distractors``).

    On samples with no extraction dict, an empty entity list, or an
    unsupported language, the stage is a no-op (existing
    ``distractor_terms`` are preserved).  A per-stage note is written
    via :func:`set_note`.

    Args:
        context_key: Manifest key holding the extraction dict produced
            by :class:`ContextualASRExtractionStage`.
        source_lang_key: Manifest key holding the per-sample source
            language.  Accepts display names (``"English"``) or ISO
            codes (``"en"``).
        default_source_lang: Fallback used when ``source_lang_key`` is
            missing or empty on a sample.
        language: Optional explicit source language (e.g. ``"English"``,
            ``"en"``, or a backend-specific code when ``g2p_backend`` is set).
            When set, used for all samples and the per-sample
            ``source_lang`` is ignored.  Use this when you know the
            entire dataset is a single language.
        g2p_backend: Backend to use for legacy/plain vocab files or when
            metadata does not pin a backend. ``"auto"`` prefers MFA when a
            model path is provided, then Phonikud/g2p-en/pypinyin/NRC/Epitran,
            and finally built-in approximate rules.
        g2p_model_path: MFA G2P model path/name, or a directory containing
            per-language MFA model archives. Used by the ``mfa`` backend.
        segments_profile_path: CLDF segments profile path for the ``segments``
            backend.
        mfa_command: MFA CLI command name or path.
        phoneme_vocab_path: Path produced by ``scripts/build_phoneme_vocab.py``.
            Either a single ``{word: [phonemes]}`` JSON file (single-language,
            applied to all samples) or a directory of ``phoneme_vocab_<lang>.json``
            files (multi-language; per-sample ``source_lang`` selects the vocab).
            Required.
        max_acoustic_distractors: Maximum acoustic distractors appended
            per sample (combined across all entities of that sample).
        max_total_distractors: Cap on the combined ``distractor_terms``
            list (semantic + acoustic) after merging.
        per_entity_top_k: Top-K candidates retained per source entity
            before cross-entity merging.
        min_npd: Lower NPD bound — entries closer than this are
            considered too-similar (typically the entity itself or a
            near-duplicate).
        max_npd: Upper NPD bound — entries farther than this are
            considered acoustically unrelated.
        notes_key: Key holding the ``additional_notes`` dict that
            :func:`set_note` writes into.
        num_workers_override: Explicit worker count for Xenna.
    """

    name: str = "AcousticDistractor"
    context_key: str = "context_asr"
    source_lang_key: str = "source_lang"
    default_source_lang: str = "English"
    language: str | None = None
    g2p_backend: str = "auto"
    g2p_model_path: str | None = None
    segments_profile_path: str | None = None
    mfa_command: str = "mfa"
    mfa_num_jobs: int = 1
    phoneme_vocab_path: str = ""
    max_acoustic_distractors: int = 8
    max_total_distractors: int = 16
    per_entity_top_k: int = 3
    min_npd: float = 0.1
    max_npd: float = 0.5
    notes_key: str = "additional_notes"
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 256

    _single_vocab: _VocabBundle | None = field(default=None, init=False, repr=False)
    _vocab_by_lang: dict[str, _VocabBundle] = field(default_factory=dict, init=False, repr=False)
    _g2p_cache: dict[tuple[tuple[str, str, str | None, str | None, str], str], list[str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _g2p_engines: dict[tuple[str, str, str | None, str | None, str], BaseG2PBackend] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _n_processed: int = field(default=0, init=False, repr=False)
    _n_appended: int = field(default=0, init=False, repr=False)

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        pass

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if not EDITDISTANCE_AVAILABLE:
            msg = "editdistance is required for AcousticDistractorStage. `pip install editdistance`."
            raise ImportError(msg)
        if not self.phoneme_vocab_path:
            msg = "AcousticDistractorStage requires phoneme_vocab_path to be set."
            raise ValueError(msg)

        vocab_path = Path(self.phoneme_vocab_path)
        if not vocab_path.exists():
            msg = f"AcousticDistractorStage: phoneme vocab path not found: {vocab_path}"
            raise FileNotFoundError(msg)

        if vocab_path.is_dir():
            files = sorted(vocab_path.glob("phoneme_vocab_*.json"))
            if not files:
                msg = f"AcousticDistractorStage: no phoneme_vocab_*.json files in directory {vocab_path}"
                raise FileNotFoundError(msg)
            for fpath in files:
                code = fpath.stem[len("phoneme_vocab_") :]
                bundle = self._load_vocab_bundle(fpath, language_hint=code)
                lang_key = self._language_key_for_bundle(bundle, code)
                if not lang_key:
                    logger.warning(
                        "%s: skipping vocab file with unmappable language code %r (%s)",
                        self.name,
                        code,
                        fpath.name,
                    )
                    continue
                self._vocab_by_lang[lang_key] = bundle
                logger.info(
                    "%s: loaded %d entries for %s (backend=%s, language=%s) from %s",
                    self.name,
                    len(bundle.items),
                    code,
                    bundle.g2p_config.backend if bundle.g2p_config else self.g2p_backend,
                    bundle.g2p_config.language if bundle.g2p_config else lang_key,
                    fpath.name,
                )
            if not self._vocab_by_lang:
                msg = f"AcousticDistractorStage: no usable phoneme_vocab_*.json files under {vocab_path}"
                raise ValueError(msg)
            logger.info(
                "%s: directory mode — %d language(s) loaded: %s",
                self.name,
                len(self._vocab_by_lang),
                ",".join(sorted(self._vocab_by_lang)),
            )
        else:
            self._single_vocab = self._load_vocab_bundle(vocab_path, language_hint=self.language)
            logger.info(
                "%s: loaded %d phoneme vocab entries from %s (language=%s)",
                self.name,
                len(self._single_vocab.items),
                vocab_path,
                self.language or "(per-sample source_lang)",
            )

        for bundle in [*self._vocab_by_lang.values(), *([self._single_vocab] if self._single_vocab else [])]:
            if bundle.g2p_config is not None:
                self._engine_for(bundle.g2p_config)

    def _load_vocab_bundle(self, path: Path, *, language_hint: Any = None) -> _VocabBundle:  # noqa: ANN401
        """Load one vocab JSON into a filtered item list plus optional metadata."""
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            msg = f"Phoneme vocab must be a JSON object; got {type(raw).__name__} ({path})."
            raise TypeError(msg)
        metadata: dict[str, Any] = {}
        vocab_raw: Any = raw
        if isinstance(raw.get("vocab"), dict):
            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            vocab_raw = raw["vocab"]
        if not isinstance(vocab_raw, dict):
            msg = f"Phoneme vocab must map word to [phones]; got {type(vocab_raw).__name__} ({path})."
            raise TypeError(msg)

        g2p_config = None
        if metadata or language_hint or self.language:
            g2p_config = build_g2p_config_from_metadata(
                metadata,
                language_hint=language_hint or self.language,
                default_backend=self.g2p_backend,
                g2p_model_path=self.g2p_model_path,
                segments_profile_path=self.segments_profile_path,
                mfa_command=self.mfa_command,
                mfa_num_jobs=self.mfa_num_jobs,
            )
        items = [
            (str(word), [str(p) for p in phonemes])
            for word, phonemes in vocab_raw.items()
            if isinstance(phonemes, list) and phonemes
        ]
        return _VocabBundle(items=items, g2p_config=g2p_config, metadata=metadata)

    @staticmethod
    def _language_key_for_bundle(bundle: _VocabBundle, fallback: Any) -> str | None:  # noqa: ANN401
        metadata = bundle.metadata
        raw_language = metadata.get("source_language") or metadata.get("normalized_language") or fallback
        if bundle.g2p_config and bundle.g2p_config.source_language:
            raw_language = bundle.g2p_config.source_language
        return normalize_g2p_language(raw_language)

    def _default_g2p_config(self, language: str) -> G2PBackendConfig | None:
        return build_g2p_config(
            self.language or language,
            backend=self.g2p_backend,
            g2p_model_path=self.g2p_model_path,
            segments_profile_path=self.segments_profile_path,
            mfa_command=self.mfa_command,
            mfa_num_jobs=self.mfa_num_jobs,
        )

    def _engine_for(self, config: G2PBackendConfig) -> BaseG2PBackend:
        key = config.cache_key
        engine = self._g2p_engines.get(key)
        if engine is None:
            engine = make_g2p_backend(config)
            self._g2p_engines[key] = engine
        return engine

    def _resolve_language(self, task: AudioTask) -> str | None:
        if self.language:
            return normalize_g2p_language(self.language)
        raw = task.data.get(self.source_lang_key) or self.default_source_lang
        return normalize_g2p_language(raw)

    def _vocab_bundle_for_language(self, language: str) -> _VocabBundle | None:
        """Return the vocab bundle for ``language``."""
        if self._vocab_by_lang:
            return self._vocab_by_lang.get(language)
        if self._single_vocab is None:
            return None
        if self._single_vocab.g2p_config is None:
            self._single_vocab.g2p_config = self._default_g2p_config(language)
        return self._single_vocab

    def _g2p(self, text: str, config: G2PBackendConfig) -> list[str]:
        key = (config.cache_key, text)
        cached = self._g2p_cache.get(key)
        if cached is not None:
            return cached
        try:
            phonemes = self._engine_for(config).phonemize(text)
        except G2PError as exc:
            logger.warning("%s: G2P failed for %r (%s/%s): %s", self.name, text, config.backend, config.language, exc)
            phonemes = []
        self._g2p_cache[key] = phonemes
        return phonemes

    def _prewarm_g2p_cache(self, tasks: list[AudioTask]) -> None:
        pending: dict[G2PBackendConfig, set[str]] = {}
        for task in tasks:
            extraction = task.data.get(self.context_key)
            if not isinstance(extraction, dict):
                continue
            fine_terms = extraction.get("fine_context_terms") or []
            if not isinstance(fine_terms, list) or not fine_terms:
                continue
            language = self._resolve_language(task)
            if not language:
                continue
            bundle = self._vocab_bundle_for_language(language)
            if bundle is None or bundle.g2p_config is None:
                continue
            for term in fine_terms:
                text = str(term)
                if (bundle.g2p_config.cache_key, text) not in self._g2p_cache:
                    pending.setdefault(bundle.g2p_config, set()).add(text)

        for config, texts_set in pending.items():
            texts = sorted(texts_set)
            try:
                phoneme_lists = self._engine_for(config).phonemize_many(texts)
            except G2PError as exc:
                logger.warning("%s: batched G2P failed for %s/%s: %s", self.name, config.backend, config.language, exc)
                phoneme_lists = [[] for _ in texts]
            for text, phonemes in zip(texts, phoneme_lists, strict=True):
                self._g2p_cache[(config.cache_key, text)] = phonemes

    def teardown(self) -> None:
        if self._n_processed:
            logger.info(
                "%s: processed %d samples, appended acoustic distractors to %d (%.1f%%)",
                self.name,
                self._n_processed,
                self._n_appended,
                100.0 * self._n_appended / self._n_processed,
            )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.context_key, self.source_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.context_key]

    def _generate_acoustic_distractors(
        self,
        fine_terms: list[str],
        existing_distractors: list[str],
        bundle: _VocabBundle,
    ) -> list[str]:
        """Return up to ``max_acoustic_distractors`` words from the vocab."""
        vocab_items = bundle.items
        if bundle.g2p_config is None:
            return []
        if not fine_terms or not vocab_items:
            return []

        excluded = {w.lower() for w in fine_terms} | {w.lower() for w in existing_distractors}

        scored: dict[str, float] = {}
        for entity in fine_terms:
            query = self._g2p(entity, bundle.g2p_config)
            if not query:
                continue
            for word, dist in _vocab_search(
                query,
                vocab_items,
                min_npd=self.min_npd,
                max_npd=self.max_npd,
                excluded_words=excluded,
                top_k=self.per_entity_top_k,
            ):
                if word.lower() in excluded:
                    continue
                prev = scored.get(word)
                if prev is None or dist < prev:
                    scored[word] = dist

        ranked = sorted(scored.items(), key=lambda kv: kv[1])
        return [w for w, _ in ranked[: self.max_acoustic_distractors]]

    def _merge_distractors(self, existing: list[str], acoustic: list[str]) -> list[str]:
        seen: set[str] = {w.lower() for w in existing}
        merged: list[str] = list(existing)
        for word in acoustic:
            if word.lower() in seen:
                continue
            seen.add(word.lower())
            merged.append(word)
            if len(merged) >= self.max_total_distractors:
                break
        return merged[: self.max_total_distractors]

    def _process_one(self, task: AudioTask) -> None:
        extraction = task.data.get(self.context_key)
        if not isinstance(extraction, dict):
            set_note(task.data, self.name, "no_extraction", self.notes_key)
            return

        fine_terms = extraction.get("fine_context_terms") or []
        if not isinstance(fine_terms, list) or not fine_terms:
            set_note(task.data, self.name, "no_fine_terms", self.notes_key)
            return

        language = self._resolve_language(task)
        if not language:
            set_note(task.data, self.name, "unsupported_language", self.notes_key)
            return

        bundle = self._vocab_bundle_for_language(language)
        if bundle is None:
            # Directory mode with no vocab file for this language.
            set_note(task.data, self.name, f"no_vocab_for_language:{language}", self.notes_key)
            return
        if bundle.g2p_config is None:
            set_note(task.data, self.name, f"no_g2p_backend_for_language:{language}", self.notes_key)
            return

        raw_existing = extraction.get("distractor_terms") or []
        existing = [str(t) for t in raw_existing] if isinstance(raw_existing, list) else []

        acoustic = self._generate_acoustic_distractors(
            [str(t) for t in fine_terms],
            existing,
            bundle,
        )

        self._n_processed += 1
        if not acoustic:
            set_note(task.data, self.name, "no_acoustic_candidates", self.notes_key)
            return

        extraction["distractor_terms"] = self._merge_distractors(existing, acoustic)
        self._n_appended += 1
        set_note(task.data, self.name, f"appended={len(acoustic)}", self.notes_key)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        self._prewarm_g2p_cache(tasks)
        for task in tasks:
            self._process_one(task)
        logger.debug("%s: batch of %d tasks", self.name, len(tasks))
        return tasks
