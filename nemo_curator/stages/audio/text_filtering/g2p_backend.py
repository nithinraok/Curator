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

"""Grapheme-to-phone backends for contextual-ASR acoustic distractors.

The acoustic distractor stage only compares strings that were encoded by the
same backend/language pair, so the returned token alphabet does not need to be a
universal IPA space.  This lets Curator use higher-quality language-specific
backends where available and permissive/rule-based fallbacks for coverage.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger

from nemo_curator.stages.audio.pipeline_utils import LANG_CODE_TO_NAME

SUPPORTED_LANGUAGE_CODES: frozenset[str] = frozenset(
    {
        "ar",
        "bg",
        "zh",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "he",
        "hi",
        "hu",
        "it",
        "ja",
        "ko",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "es",
        "sv",
        "th",
        "uk",
    }
)

SUPPORTED_LANGUAGE_NAMES: dict[str, str] = {
    code: LANG_CODE_TO_NAME[code] for code in sorted(SUPPORTED_LANGUAGE_CODES) if code in LANG_CODE_TO_NAME
}

_LANG_NAME_TO_CODE: dict[str, str] = {name.lower(): code for code, name in LANG_CODE_TO_NAME.items()}

_LANG_ALIASES: dict[str, str] = {
    "arabic": "ar",
    "bulgarian": "bg",
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "chinese": "zh",
    "cmn": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "hrv": "hr",
    "ces": "cs",
    "cze": "cs",
    "dan": "da",
    "nld": "nl",
    "dut": "nl",
    "eng": "en",
    "en-us": "en",
    "en-gb": "en",
    "est": "et",
    "fin": "fi",
    "fra": "fr",
    "fre": "fr",
    "fr-fr": "fr",
    "deu": "de",
    "ger": "de",
    "ell": "el",
    "gre": "el",
    "heb": "he",
    "iw": "he",
    "hin": "hi",
    "hun": "hu",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "lav": "lv",
    "lit": "lt",
    "mlt": "mt",
    "pol": "pl",
    "por": "pt",
    "pt-br": "pt",
    "pt-pt": "pt",
    "ron": "ro",
    "rum": "ro",
    "rus": "ru",
    "slk": "sk",
    "slo": "sk",
    "slv": "sl",
    "spa": "es",
    "swe": "sv",
    "tha": "th",
    "ukr": "uk",
}

_EPITRAN_CODES: dict[str, str] = {
    "ar": "ara-Arab",
    "zh": "cmn-Hans",
    "hr": "hrv-Latn",
    "cs": "ces-Latn",
    "nl": "nld-Latn",
    "en": "eng-Latn",
    "et": "est-Latn",
    "fi": "fin-Latn",
    "fr": "fra-Latn",
    "de": "deu-Latn",
    "hi": "hin-Deva",
    "hu": "hun-Latn",
    "it": "ita-Latn",
    "ja": "jpn-Jpan",
    "ko": "kor-Hang",
    "lv": "lav-Latn",
    "lt": "lit-Latn",
    "mt": "mlt-Latn",
    "pl": "pol-Latn",
    "pt": "por-Latn",
    "ro": "ron-Latn",
    "ru": "rus-Cyrl",
    "sl": "slv-Latn",
    "es": "spa-Latn",
    "sv": "swe-Latn",
    "th": "tha-Thai",
    "uk": "ukr-Cyrl",
}

_MFA_PREFERRED_MODEL_NAMES: dict[str, tuple[str, ...]] = {
    "bg": ("bulgarian_mfa",),
    "zh": ("mandarin_china_mfa", "mandarin_china_pinyin_mfa"),
    "hr": ("croatian_mfa", "serbocroatian_croatian_mfa"),
    "cs": ("czech_mfa",),
    "en": ("english_us_mfa", "english_us_arpa", "english_uk_mfa"),
    "fr": ("french_mfa",),
    "de": ("german_mfa",),
    "ja": ("japanese_mfa", "japanese_katakana_mfa"),
    "ko": ("korean_mfa", "korean_jamo_mfa"),
    "pl": ("polish_mfa",),
    "pt": ("portuguese_brazil_mfa", "portuguese_portugal_mfa"),
    "ru": ("russian_mfa",),
    "es": ("spanish_spain_mfa", "spanish_latin_america_mfa"),
    "sv": ("swedish_mfa",),
    "th": ("thai_mfa",),
    "uk": ("ukrainian_mfa",),
}

_MFA_MODEL_LANGUAGE_CODES: frozenset[str] = frozenset(_MFA_PREFERRED_MODEL_NAMES)

_NRC_G2P_CODES: dict[str, str] = {}

_BACKEND_ALIASES: dict[str, str] = {
    "auto": "auto",
    "mfa": "mfa",
    "mfa_cli": "mfa",
    "montreal_forced_aligner": "mfa",
    "phonikud": "phonikud",
    "hebrew": "phonikud",
    "epitran": "epitran",
    "pypinyin": "pypinyin",
    "pinyin": "pypinyin",
    "g2p_en": "g2p_en",
    "g2p-en": "g2p_en",
    "english": "g2p_en",
    "nrc_g2p": "nrc_g2p",
    "nrc-ilt": "nrc_g2p",
    "nrc": "nrc_g2p",
    "segments": "segments",
    "rules": "rules",
    "fallback": "rules",
}

_IPA_ATTACHERS: frozenset[str] = frozenset(
    {
        "\u02d0",  # triangular colon
        "\u02d1",  # half triangular colon
        "\u02b0",
        "\u02b2",
        "\u02b7",
        "\u02e0",
        "\u02e4",
        "\u02de",
        "\u0330",
        "\u0331",
        "\u0339",
        "\u033a",
        "\u033b",
        "\u033c",
        "\u033d",
        "\u0361",
        "\u035c",
    }
)

_STRESS_MARKS: frozenset[str] = frozenset({"\u02c8", "\u02cc"})
_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)


class G2PError(RuntimeError):
    """Raised when a configured G2P backend cannot run."""


@dataclass(frozen=True)
class G2PBackendConfig:
    """Resolved backend configuration used for both vocab and query G2P."""

    backend: str
    language: str
    source_language: str | None = None
    model_path: str | None = None
    segments_profile_path: str | None = None
    mfa_command: str = "mfa"
    mfa_num_jobs: int = 1

    @property
    def cache_key(self) -> tuple[str, str, str | None, str | None, str]:
        return (self.backend, self.language, self.model_path, self.segments_profile_path, self.mfa_command)


class BaseG2PBackend:
    """Small interface shared by all G2P backends."""

    def __init__(self, config: G2PBackendConfig) -> None:
        self.config = config

    def phonemize(self, text: str) -> list[str]:
        return self.phonemize_many([text])[0]

    def phonemize_many(self, texts: list[str]) -> list[list[str]]:
        return [self._phonemize_one(text) for text in texts]

    def _phonemize_one(self, text: str) -> list[str]:
        raise NotImplementedError


def normalize_g2p_backend(value: str | None) -> str:
    """Normalize a backend name/alias."""
    if value is None:
        return "auto"
    key = str(value).strip().lower().replace(" ", "_")
    return _BACKEND_ALIASES.get(key, key)


def normalize_g2p_language(value: Any) -> str | None:  # noqa: ANN401
    """Normalize a display name, ISO code, or common backend code to ISO-639-1."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower().replace("_", "-")
    if lowered in SUPPORTED_LANGUAGE_CODES:
        return lowered
    if lowered in _LANG_ALIASES:
        return _LANG_ALIASES[lowered]
    if lowered in _LANG_NAME_TO_CODE:
        code = _LANG_NAME_TO_CODE[lowered]
        return code if code in SUPPORTED_LANGUAGE_CODES else None
    if "-" in lowered:
        prefix = lowered.split("-", 1)[0]
        if prefix in SUPPORTED_LANGUAGE_CODES:
            return prefix
        if prefix in _LANG_ALIASES:
            return _LANG_ALIASES[prefix]
    return None


def build_g2p_config(  # noqa: PLR0911, PLR0913
    language: Any,  # noqa: ANN401
    *,
    backend: str = "auto",
    g2p_model_path: str | None = None,
    segments_profile_path: str | None = None,
    mfa_command: str = "mfa",
    mfa_num_jobs: int = 1,
) -> G2PBackendConfig | None:
    """Resolve a requested backend/language into a concrete G2P config."""
    normalized_backend = normalize_g2p_backend(backend)
    iso = normalize_g2p_language(language)
    raw_language = str(language).strip() if language else ""

    if normalized_backend != "auto":
        return _build_explicit_g2p_config(
            normalized_backend,
            iso,
            raw_language,
            g2p_model_path=g2p_model_path,
            segments_profile_path=segments_profile_path,
            mfa_command=mfa_command,
            mfa_num_jobs=mfa_num_jobs,
        )

    if not iso:
        return None

    if iso == "he" and _package_available("phonikud"):
        return G2PBackendConfig(backend="phonikud", language="he", source_language=iso)
    if iso == "en" and _package_available("g2p_en"):
        return G2PBackendConfig(backend="g2p_en", language="en", source_language=iso)

    mfa_model = _resolve_mfa_model_path(g2p_model_path, iso)
    if mfa_model and iso in _MFA_MODEL_LANGUAGE_CODES:
        return G2PBackendConfig(
            backend="mfa",
            language=iso,
            source_language=iso,
            model_path=mfa_model,
            mfa_command=mfa_command,
            mfa_num_jobs=mfa_num_jobs,
        )

    if iso == "zh" and _package_available("pypinyin"):
        return G2PBackendConfig(backend="pypinyin", language="zh", source_language=iso)
    if iso in _NRC_G2P_CODES and _nrc_g2p_has_path(_NRC_G2P_CODES[iso]):
        return G2PBackendConfig(backend="nrc_g2p", language=_NRC_G2P_CODES[iso], source_language=iso)
    if iso in _EPITRAN_CODES and _package_available("epitran"):
        return G2PBackendConfig(backend="epitran", language=_EPITRAN_CODES[iso], source_language=iso)
    return G2PBackendConfig(backend="rules", language=iso, source_language=iso)


def build_g2p_config_from_metadata(  # noqa: PLR0913
    metadata: dict[str, Any],
    *,
    language_hint: Any = None,  # noqa: ANN401
    default_backend: str = "auto",
    g2p_model_path: str | None = None,
    segments_profile_path: str | None = None,
    mfa_command: str = "mfa",
    mfa_num_jobs: int = 1,
) -> G2PBackendConfig | None:
    """Resolve a G2P config from vocab metadata, falling back to stage settings."""
    backend = str(metadata.get("g2p_backend") or metadata.get("backend") or default_backend)
    language = (
        metadata.get("g2p_language")
        or metadata.get("language")
        or metadata.get("source_language")
        or metadata.get("normalized_language")
        or language_hint
    )
    model_path = str(metadata.get("g2p_model_path") or g2p_model_path or "") or None
    profile_path = str(metadata.get("segments_profile_path") or segments_profile_path or "") or None
    return build_g2p_config(
        language,
        backend=backend,
        g2p_model_path=model_path,
        segments_profile_path=profile_path,
        mfa_command=str(metadata.get("mfa_command") or mfa_command),
        mfa_num_jobs=int(metadata.get("mfa_num_jobs") or mfa_num_jobs),
    )


def make_g2p_backend(config: G2PBackendConfig) -> BaseG2PBackend:
    """Instantiate the concrete backend for ``config``."""
    backend = normalize_g2p_backend(config.backend)
    if backend == "mfa":
        return MfaCliBackend(config)
    if backend == "phonikud":
        return PhonikudBackend(config)
    if backend == "g2p_en":
        return G2pEnBackend(config)
    if backend == "pypinyin":
        return PypinyinBackend(config)
    if backend == "nrc_g2p":
        return NrcG2PBackend(config)
    if backend == "epitran":
        return EpitranBackend(config)
    if backend == "segments":
        return SegmentsBackend(config)
    if backend == "rules":
        return RulesBackend(config)
    msg = f"Unsupported G2P backend: {config.backend}"
    raise G2PError(msg)


def g2p_metadata(config: G2PBackendConfig, *, input_language: str | None = None) -> dict[str, Any]:
    """Serialize vocab metadata for a resolved backend config."""
    model_path = config.model_path
    if config.backend == "mfa" and model_path:
        path = Path(model_path)
        if path.suffix == ".zip":
            model_path = path.stem
    mfa_command = None
    if config.backend == "mfa" and Path(config.mfa_command).name == config.mfa_command:
        mfa_command = config.mfa_command
    return {
        "format": "curator_acoustic_distractor_vocab_v2",
        "g2p_backend": config.backend,
        "g2p_language": config.language,
        "source_language": config.source_language or normalize_g2p_language(input_language) or input_language,
        "g2p_model_path": model_path,
        "segments_profile_path": config.segments_profile_path,
        "mfa_command": mfa_command,
    }


def _build_explicit_g2p_config(  # noqa: PLR0911, PLR0913
    backend: str,
    iso: str | None,
    raw_language: str,
    *,
    g2p_model_path: str | None,
    segments_profile_path: str | None,
    mfa_command: str,
    mfa_num_jobs: int,
) -> G2PBackendConfig | None:
    source_language = iso
    if backend == "mfa":
        model_path = _resolve_mfa_model_path(g2p_model_path, iso or raw_language)
        language = iso or raw_language
        return G2PBackendConfig(
            backend="mfa",
            language=language,
            source_language=source_language,
            model_path=model_path,
            mfa_command=mfa_command,
            mfa_num_jobs=mfa_num_jobs,
        )
    if backend == "epitran":
        language = _EPITRAN_CODES.get(iso or "", raw_language)
        return G2PBackendConfig(backend="epitran", language=language, source_language=source_language)
    if backend == "phonikud":
        return G2PBackendConfig(backend="phonikud", language="he", source_language=source_language or "he")
    if backend == "g2p_en":
        return G2PBackendConfig(backend="g2p_en", language="en", source_language=source_language or "en")
    if backend == "pypinyin":
        return G2PBackendConfig(backend="pypinyin", language="zh", source_language=source_language or "zh")
    if backend == "nrc_g2p":
        language = _NRC_G2P_CODES.get(iso or "", raw_language or "dan")
        return G2PBackendConfig(backend="nrc_g2p", language=language, source_language=source_language)
    if backend == "segments":
        language = iso or raw_language
        return G2PBackendConfig(
            backend="segments",
            language=language,
            source_language=source_language,
            segments_profile_path=segments_profile_path,
        )
    if backend == "rules":
        language = iso or raw_language
        return G2PBackendConfig(backend="rules", language=language, source_language=source_language)
    msg = f"Unsupported G2P backend: {backend}"
    raise G2PError(msg)


def _package_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


@lru_cache(maxsize=None)
def _nrc_g2p_has_path(language: str) -> bool:
    try:
        from g2p import make_g2p

        make_g2p(language, "ipa")
    except Exception:  # noqa: BLE001
        return False
    return True


def _resolve_mfa_model_path(model_path: str | None, language: str | None) -> str | None:
    if not model_path:
        return None
    path = Path(model_path)
    if not path.is_dir():
        return str(model_path)

    iso = normalize_g2p_language(language) or str(language or "").lower()
    display = SUPPORTED_LANGUAGE_NAMES.get(iso, iso).lower().replace(" ", "_")
    preferred = _MFA_PREFERRED_MODEL_NAMES.get(iso, ())
    candidates = [
        *(f"{name}.zip" for name in preferred),
        f"mfa_g2p_{iso}.zip",
        f"g2p_{iso}.zip",
        f"{iso}.zip",
        f"{display}_mfa_g2p.zip",
        f"{display}_mfa.zip",
        f"{display}.zip",
    ]
    for candidate in candidates:
        candidate_path = path / candidate
        if candidate_path.exists():
            return str(candidate_path)
    for pattern in (f"*{display}*g2p*.zip", f"*{display}*mfa*.zip", f"*{iso}*g2p*.zip", f"*{iso}*.zip"):
        matches = sorted(path.glob(pattern))
        if matches:
            return str(matches[0])
    return None


def _strip_punctuation_token(token: str) -> str:
    return "".join(ch for ch in token if not unicodedata.category(ch).startswith("P"))


def _ipa_to_tokens(text: str) -> list[str]:
    """Tokenize an IPA-like string without requiring a GPL tokenizer."""
    tokens: list[str] = []
    current = ""
    for char in text:
        if char in _STRESS_MARKS:
            continue
        category = unicodedata.category(char)
        if char.isspace() or category.startswith("P") or category.startswith("S"):
            if current:
                tokens.append(current)
                current = ""
            continue
        if unicodedata.combining(char) or char in _IPA_ATTACHERS:
            if current:
                current += char
            continue
        if current:
            tokens.append(current)
        current = char
    if current:
        tokens.append(current)
    return tokens


def _normalize_latin_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _word_tokens(text: str) -> list[str]:
    return [_strip_punctuation_token(match.group(0).lower()) for match in _WORD_RE.finditer(text)]


class RulesBackend(BaseG2PBackend):
    """Curator-owned approximate fallback for coverage and tests."""

    _CYRILLIC: ClassVar[dict[str, str]] = {
        "\u0430": "a",
        "\u0431": "b",
        "\u0432": "v",
        "\u0433": "g",
        "\u0434": "d",
        "\u0435": "e",
        "\u0451": "yo",
        "\u0436": "zh",
        "\u0437": "z",
        "\u0438": "i",
        "\u0439": "j",
        "\u043a": "k",
        "\u043b": "l",
        "\u043c": "m",
        "\u043d": "n",
        "\u043e": "o",
        "\u043f": "p",
        "\u0440": "r",
        "\u0441": "s",
        "\u0442": "t",
        "\u0443": "u",
        "\u0444": "f",
        "\u0445": "h",
        "\u0446": "ts",
        "\u0447": "ch",
        "\u0448": "sh",
        "\u0449": "sht",
        "\u044a": "",
        "\u044b": "y",
        "\u044c": "",
        "\u044d": "e",
        "\u044e": "yu",
        "\u044f": "ya",
        "\u0454": "ye",
        "\u0456": "i",
        "\u0457": "yi",
        "\u0491": "g",
    }
    _GREEK: ClassVar[dict[str, str]] = {
        "\u03b1": "a",
        "\u03b2": "v",
        "\u03b3": "g",
        "\u03b4": "d",
        "\u03b5": "e",
        "\u03b6": "z",
        "\u03b7": "i",
        "\u03b8": "th",
        "\u03b9": "i",
        "\u03ba": "k",
        "\u03bb": "l",
        "\u03bc": "m",
        "\u03bd": "n",
        "\u03be": "ks",
        "\u03bf": "o",
        "\u03c0": "p",
        "\u03c1": "r",
        "\u03c3": "s",
        "\u03c2": "s",
        "\u03c4": "t",
        "\u03c5": "i",
        "\u03c6": "f",
        "\u03c7": "h",
        "\u03c8": "ps",
        "\u03c9": "o",
    }
    _HEBREW: ClassVar[dict[str, str]] = {
        "\u05d0": "",
        "\u05d1": "b",
        "\u05d2": "g",
        "\u05d3": "d",
        "\u05d4": "h",
        "\u05d5": "v",
        "\u05d6": "z",
        "\u05d7": "h",
        "\u05d8": "t",
        "\u05d9": "y",
        "\u05da": "k",
        "\u05db": "k",
        "\u05dc": "l",
        "\u05dd": "m",
        "\u05de": "m",
        "\u05df": "n",
        "\u05e0": "n",
        "\u05e1": "s",
        "\u05e2": "",
        "\u05e3": "f",
        "\u05e4": "p",
        "\u05e5": "ts",
        "\u05e6": "ts",
        "\u05e7": "k",
        "\u05e8": "r",
        "\u05e9": "sh",
        "\u05ea": "t",
    }

    def _phonemize_one(self, text: str) -> list[str]:
        language = normalize_g2p_language(self.config.language) or self.config.language
        if language in {"bg", "ru", "uk"}:
            return self._map_chars(text, self._CYRILLIC)
        if language == "el":
            return self._map_chars(text, self._GREEK)
        if language == "he":
            return self._map_chars(text, self._HEBREW)
        return self._latinish_tokens(text)

    @classmethod
    def _map_chars(cls, text: str, mapping: dict[str, str]) -> list[str]:
        tokens: list[str] = []
        for char in unicodedata.normalize("NFKD", text.lower()):
            if unicodedata.combining(char) or char.isspace() or unicodedata.category(char).startswith("P"):
                continue
            mapped = mapping.get(char)
            if mapped is None:
                if char.isalnum():
                    tokens.append(char)
                continue
            if mapped:
                tokens.append(mapped)
        return tokens

    @staticmethod
    def _latinish_tokens(text: str) -> list[str]:
        normalized = _normalize_latin_text(text)
        tokens: list[str] = []
        for word in _word_tokens(normalized):
            if not word:
                continue
            idx = 0
            while idx < len(word):
                tri = word[idx : idx + 3]
                duo = word[idx : idx + 2]
                if tri in {"sch", "dzs"}:
                    tokens.append(tri)
                    idx += 3
                elif duo in {"ch", "sh", "zh", "th", "ph", "kh", "ng", "ny", "ty", "gy", "sz", "zs", "cs", "dz"}:
                    tokens.append(duo)
                    idx += 2
                else:
                    tokens.append(word[idx])
                    idx += 1
        return tokens


class PhonikudBackend(BaseG2PBackend):
    def __init__(self, config: G2PBackendConfig) -> None:
        super().__init__(config)
        try:
            from phonikud import phonemize
        except ImportError as exc:
            msg = "phonikud is required for the Hebrew Phonikud G2P backend (`pip install phonikud`)."
            raise G2PError(msg) from exc
        self._phonemize = phonemize

    def _phonemize_one(self, text: str) -> list[str]:
        output = self._phonemize(text)
        return _ipa_to_tokens(str(output))


class G2pEnBackend(BaseG2PBackend):
    def __init__(self, config: G2PBackendConfig) -> None:
        super().__init__(config)
        try:
            from g2p_en import G2p
        except ImportError as exc:
            msg = "g2p-en is required for the English G2P backend (`pip install g2p-en`)."
            raise G2PError(msg) from exc
        self._g2p = G2p()

    def _phonemize_one(self, text: str) -> list[str]:
        raw_tokens = self._g2p(text)
        tokens: list[str] = []
        for token in raw_tokens:
            cleaned = re.sub(r"\d", "", str(token).strip())
            if cleaned and cleaned not in {" ", ".", ",", ";", ":", "!", "?"}:
                tokens.append(cleaned)
        return tokens


class PypinyinBackend(BaseG2PBackend):
    def __init__(self, config: G2PBackendConfig) -> None:
        super().__init__(config)
        try:
            from pypinyin import Style, lazy_pinyin
        except ImportError as exc:
            msg = "pypinyin is required for the Chinese pinyin G2P backend (`pip install pypinyin`)."
            raise G2PError(msg) from exc
        self._style = Style.TONE3
        self._lazy_pinyin = lazy_pinyin

    def _phonemize_one(self, text: str) -> list[str]:
        tokens = self._lazy_pinyin(text, style=self._style, neutral_tone_with_five=True, errors="ignore")
        return [str(token).strip().lower() for token in tokens if str(token).strip()]


class EpitranBackend(BaseG2PBackend):
    def __init__(self, config: G2PBackendConfig) -> None:
        super().__init__(config)
        try:
            import epitran
        except ImportError as exc:
            msg = "epitran is required for the Epitran G2P backend (`pip install epitran`)."
            raise G2PError(msg) from exc
        self._epitran = epitran.Epitran(config.language)

    def _phonemize_one(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in _word_tokens(text):
            tokens.extend(_ipa_to_tokens(self._epitran.transliterate(word)))
        if tokens:
            return tokens
        return _ipa_to_tokens(self._epitran.transliterate(text))


class NrcG2PBackend(BaseG2PBackend):
    def __init__(self, config: G2PBackendConfig) -> None:
        super().__init__(config)
        try:
            from g2p import make_g2p
        except ImportError as exc:
            msg = "NRC-ILT g2p is required for this G2P backend (`pip install g2p`)."
            raise G2PError(msg) from exc
        try:
            self._transducer = make_g2p(config.language, "ipa")
        except Exception as exc:  # noqa: BLE001
            msg = f"NRC-ILT g2p has no path from {config.language!r} to IPA."
            raise G2PError(msg) from exc

    def _phonemize_one(self, text: str) -> list[str]:
        output = self._transducer(text)
        if hasattr(output, "output_string"):
            return _ipa_to_tokens(str(output.output_string))
        return _ipa_to_tokens(str(output))


class SegmentsBackend(BaseG2PBackend):
    def __init__(self, config: G2PBackendConfig) -> None:
        super().__init__(config)
        if not config.segments_profile_path:
            msg = "segments backend requires segments_profile_path."
            raise G2PError(msg)
        try:
            from segments import Tokenizer
        except ImportError as exc:
            msg = "segments is required for the segments backend (`pip install segments`)."
            raise G2PError(msg) from exc
        self._tokenizer = Tokenizer(config.segments_profile_path)

    def _phonemize_one(self, text: str) -> list[str]:
        if hasattr(self._tokenizer, "transform"):
            output = self._tokenizer.transform(text)
        else:
            output = self._tokenizer(text)
        output_text = str(output)
        if " " in output_text:
            return [token for token in output_text.split() if token]
        return _ipa_to_tokens(output_text)


class MfaCliBackend(BaseG2PBackend):
    def __init__(self, config: G2PBackendConfig) -> None:
        super().__init__(config)
        if not config.model_path:
            msg = "MFA backend requires g2p_model_path pointing to an MFA G2P model."
            raise G2PError(msg)
        command = config.mfa_command
        if Path(command).name == command and shutil.which(command) is None:
            msg = f"MFA command not found: {command}. Install Montreal Forced Aligner or set mfa_command."
            raise G2PError(msg)

    def phonemize_many(self, texts: list[str]) -> list[list[str]]:
        text_words = [self._text_to_words(text) for text in texts]
        unique_words = sorted({word for words in text_words for word in words})
        pronunciations = self._run_mfa(unique_words) if unique_words else {}
        fallback = RulesBackend(
            G2PBackendConfig(backend="rules", language=self.config.source_language or self.config.language)
        )

        output: list[list[str]] = []
        for words, text in zip(text_words, texts, strict=True):
            tokens: list[str] = []
            for word in words:
                tokens.extend(pronunciations.get(word) or fallback.phonemize(word))
            if not tokens:
                tokens = fallback.phonemize(text)
            output.append(tokens)
        return output

    @staticmethod
    def _text_to_words(text: str) -> list[str]:
        words = _word_tokens(text)
        if words:
            return words
        stripped = text.strip().lower()
        return [stripped] if stripped else []

    def _run_mfa(self, words: list[str]) -> dict[str, list[str]]:
        with tempfile.TemporaryDirectory(prefix="curator_mfa_g2p_") as tmpdir:
            tmp_path = Path(tmpdir)
            work_path = tmp_path / "mfa_work"
            work_path.mkdir()
            input_path = tmp_path / "words.txt"
            output_path = tmp_path / "pronunciations.dict"
            input_path.write_text("\n".join(words), encoding="utf-8")
            command = [
                self.config.mfa_command,
                "g2p",
                "--quiet",
                "--clean",
                "--no_final_clean",
                "--strict_graphemes",
                "--temporary_directory",
                str(work_path),
                "--num_jobs",
                str(max(1, self.config.mfa_num_jobs)),
                str(input_path),
                str(self.config.model_path),
                str(output_path),
            ]
            env = os.environ.copy()
            command_dir = str(Path(self.config.mfa_command).parent)
            if command_dir and command_dir != ".":
                env["PATH"] = command_dir + os.pathsep + env.get("PATH", "")
            result = subprocess.run(command, capture_output=True, text=True, check=False, env=env)  # noqa: S603, S607
            if result.returncode != 0:
                msg = f"MFA G2P failed for {self.config.language}: {result.stderr.strip() or result.stdout.strip()}"
                raise G2PError(msg)
            if not output_path.exists():
                logger.warning("MFA G2P produced no output file for %s", self.config.language)
                return {}
            return self._parse_dictionary(output_path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_dictionary(text: str) -> dict[str, list[str]]:
        pronunciations: dict[str, list[str]] = {}
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            phones = parts[1:]
            pronunciations.setdefault(word, phones)
        return pronunciations
