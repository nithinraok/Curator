#!/usr/bin/env python3
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

"""Precompute a phone-token vocabulary for AcousticDistractorStage.

Reads a user-provided vocabulary file (one word/phrase per JSON array
entry, or one entry per line if the input is plain text), runs each entry
through a configured G2P backend for the requested language, and writes a
JSON payload consumed by
``nemo_curator/stages/audio/text_filtering/acoustic_distractor.py``.

The default output includes metadata so runtime query terms are encoded
with the same backend/language as the vocabulary:

    {"metadata": {...}, "vocab": {"word": ["phone", ...]}}

Use ``--plain_json`` only when a downstream tool requires the old
``{"word": ["phone", ...]}`` shape.

Usage
-----
    python scripts/build_phoneme_vocab.py \\
        --vocab vocab.json \\
        --language English \\
        --backend auto \\
        --output phoneme_vocab_en.json

For MFA-backed vocabularies, pass an MFA G2P model name/path or a directory
containing per-language model archives:

    python scripts/build_phoneme_vocab.py \\
        --vocab vocab_en.txt \\
        --language en \\
        --backend mfa \\
        --g2p_model_path english_mfa \\
        --output phoneme_vocab_en.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemo_curator.stages.audio.text_filtering.g2p_backend import (
    G2PError,
    build_g2p_config,
    g2p_metadata,
    make_g2p_backend,
)


def _load_vocab(vocab_path: Path) -> list[str]:
    """Load vocab from JSON (list of strings) or plain text (one per line)."""
    raw = vocab_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if isinstance(parsed, list):
        return [str(w).strip() for w in parsed if str(w).strip()]
    if isinstance(parsed, dict):
        return [str(w).strip() for w in parsed if str(w).strip()]
    msg = f"Unsupported vocab JSON structure: top-level {type(parsed).__name__}; expected list or object."
    raise ValueError(msg)


def _dedupe_preserving_order(words: list[str], *, lowercase: bool) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        key = w.lower() if lowercase else w
        if key in seen:
            continue
        seen.add(key)
        out.append(key if lowercase else w)
    return out


def _phonemize_batch(  # noqa: PLR0913
    words: list[str],
    *,
    backend: str,
    language: str,
    g2p_model_path: str | None,
    segments_profile_path: str | None,
    mfa_command: str,
    n_jobs: int,
    batch_size: int,
) -> tuple[list[list[str]], dict[str, object]]:
    """Run the configured G2P backend and return per-word phone token lists."""
    config = build_g2p_config(
        language,
        backend=backend,
        g2p_model_path=g2p_model_path,
        segments_profile_path=segments_profile_path,
        mfa_command=mfa_command,
        mfa_num_jobs=n_jobs,
    )
    if config is None:
        msg = f"Could not resolve a G2P backend for language={language!r}, backend={backend!r}."
        raise ValueError(msg)

    g2p = make_g2p_backend(config)
    results: list[list[str]] = []
    total = len(words)
    for start in range(0, total, batch_size):
        chunk = words[start : start + batch_size]
        try:
            phone_lists = g2p.phonemize_many(chunk)
        except G2PError as exc:
            msg = f"G2P failed for backend={config.backend}, language={config.language}: {exc}"
            raise RuntimeError(msg) from exc
        results.extend(phone_lists)
        done = min(start + batch_size, total)
        print(f"[g2p:{config.backend}] {done}/{total}", file=sys.stderr, flush=True)
    return results, g2p_metadata(config, input_language=language)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a precomputed phone vocabulary for AcousticDistractorStage.")
    ap.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to the input vocab file (JSON array/object of strings, or plain text with one entry per line).",
    )
    ap.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language name/code for G2P routing (e.g. English, en, fr, zh, Hebrew).",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default="auto",
        help=(
            "G2P backend: auto, mfa, phonikud, g2p-en, pypinyin, nrc_g2p, epitran, segments, or rules. "
            "auto prefers MFA when --g2p_model_path resolves for the language."
        ),
    )
    ap.add_argument(
        "--g2p_model_path",
        type=str,
        default=None,
        help="MFA G2P model name/path, or a directory containing per-language MFA model archives.",
    )
    ap.add_argument(
        "--segments_profile_path",
        type=str,
        default=None,
        help="CLDF segments profile path, required only when --backend segments.",
    )
    ap.add_argument(
        "--mfa_command",
        type=str,
        default="mfa",
        help="Montreal Forced Aligner CLI command name/path used by --backend mfa.",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the phone-vocabulary JSON output.",
    )
    ap.add_argument(
        "--lowercase",
        action="store_true",
        default=True,
        help="Lowercase vocab entries before G2P and dedup (default: True).",
    )
    ap.add_argument(
        "--no-lowercase",
        action="store_false",
        dest="lowercase",
        help="Preserve original case (useful for languages without case, e.g. Chinese, Japanese, Thai).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Number of entries sent to the G2P backend per call.",
    )
    ap.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Backend worker count where supported (currently passed to MFA).",
    )
    ap.add_argument(
        "--plain_json",
        action="store_true",
        default=False,
        help="Write legacy {word: [phones]} JSON instead of metadata-wrapped JSON.",
    )
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    vocab_path = Path(args.vocab)
    output_path = Path(args.output)

    if not vocab_path.exists():
        msg = f"Vocab file not found: {vocab_path}"
        raise FileNotFoundError(msg)

    raw_words = _load_vocab(vocab_path)
    if not raw_words:
        msg = f"No vocabulary entries found in {vocab_path}."
        raise ValueError(msg)

    words = _dedupe_preserving_order(raw_words, lowercase=args.lowercase)
    print(
        f"[vocab] loaded {len(raw_words)} entries, {len(words)} unique after dedup "
        f"(lowercase={args.lowercase})",
        file=sys.stderr,
        flush=True,
    )

    phoneme_lists, metadata = _phonemize_batch(
        words,
        backend=args.backend,
        language=args.language,
        g2p_model_path=args.g2p_model_path,
        segments_profile_path=args.segments_profile_path,
        mfa_command=args.mfa_command,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
    )

    vocab_out: dict[str, list[str]] = {}
    n_empty = 0
    for word, phonemes in zip(words, phoneme_lists, strict=True):
        if not phonemes:
            n_empty += 1
            continue
        vocab_out[word] = phonemes

    metadata.update(
        {
            "input_vocab_path": vocab_path.name,
            "lowercase": args.lowercase,
            "input_entries": len(raw_words),
            "deduped_entries": len(words),
            "written_entries": len(vocab_out),
            "empty_entries": n_empty,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = vocab_out if args.plain_json else {"metadata": metadata, "vocab": vocab_out}
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)

    print(
        f"[vocab] wrote {len(vocab_out)} entries to {output_path} "
        f"(skipped {n_empty} with empty G2P output)",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
