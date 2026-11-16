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

import json
from pathlib import Path

from nemo_curator.stages.audio.text_filtering.acoustic_distractor import AcousticDistractorStage
from nemo_curator.stages.audio.text_filtering.g2p_backend import (
    G2PBackendConfig,
    SUPPORTED_LANGUAGE_NAMES,
    build_g2p_config,
    g2p_metadata,
)
from nemo_curator.tasks import AudioTask


def test_auto_backend_resolves_all_target_languages() -> None:
    for code, display_name in SUPPORTED_LANGUAGE_NAMES.items():
        config = build_g2p_config(display_name, backend="auto")
        assert config is not None, display_name
        assert config.source_language == code


def test_stage_uses_metadata_wrapped_vocab(tmp_path: Path) -> None:
    vocab_path = tmp_path / "phoneme_vocab_en.json"
    vocab_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "g2p_backend": "rules",
                    "g2p_language": "en",
                    "source_language": "en",
                },
                "vocab": {
                    "hello": ["h", "e", "l", "l", "o"],
                    "hullo": ["h", "u", "l", "l", "o"],
                    "world": ["w", "o", "r", "l", "d"],
                },
            }
        ),
        encoding="utf-8",
    )

    stage = AcousticDistractorStage(
        phoneme_vocab_path=str(vocab_path),
        max_acoustic_distractors=2,
        min_npd=0.1,
        max_npd=0.5,
    )
    stage.setup()

    task = AudioTask(
        data={
            "source_lang": "English",
            "context_asr": {
                "fine_context_terms": ["hello"],
                "distractor_terms": ["existing"],
            },
        }
    )

    result = stage.process(task)

    assert result.data["context_asr"]["distractor_terms"] == ["existing", "hullo"]
    assert result.data["additional_notes"]["AcousticDistractor"] == "appended=1"


def test_stage_supports_legacy_plain_vocab_with_stage_backend(tmp_path: Path) -> None:
    vocab_path = tmp_path / "plain_vocab.json"
    vocab_path.write_text(
        json.dumps(
            {
                "hello": ["h", "e", "l", "l", "o"],
                "hullo": ["h", "u", "l", "l", "o"],
            }
        ),
        encoding="utf-8",
    )

    stage = AcousticDistractorStage(
        phoneme_vocab_path=str(vocab_path),
        language="en",
        g2p_backend="rules",
        max_acoustic_distractors=1,
        min_npd=0.1,
        max_npd=0.5,
    )
    stage.setup()

    task = AudioTask(
        data={
            "source_lang": "English",
            "context_asr": {
                "fine_context_terms": ["hello"],
                "distractor_terms": [],
            },
        }
    )

    result = stage.process(task)

    assert result.data["context_asr"]["distractor_terms"] == ["hullo"]


def test_stage_directory_mode_uses_sample_language(tmp_path: Path) -> None:
    (tmp_path / "phoneme_vocab_en.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "g2p_backend": "rules",
                    "g2p_language": "en",
                    "source_language": "en",
                },
                "vocab": {"hullo": ["h", "u", "l", "l", "o"]},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "phoneme_vocab_fr.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "g2p_backend": "rules",
                    "g2p_language": "fr",
                    "source_language": "fr",
                },
                "vocab": {"bonjour": ["b", "o", "n", "j", "o", "u", "r"]},
            }
        ),
        encoding="utf-8",
    )

    stage = AcousticDistractorStage(
        phoneme_vocab_path=str(tmp_path),
        max_acoustic_distractors=2,
        min_npd=0.1,
        max_npd=0.5,
    )
    stage.setup()

    task = AudioTask(
        data={
            "source_lang": "English",
            "context_asr": {
                "fine_context_terms": ["hello"],
                "distractor_terms": [],
            },
        }
    )

    result = stage.process(task)

    assert result.data["context_asr"]["distractor_terms"] == ["hullo"]


def test_mfa_metadata_uses_portable_references() -> None:
    config = G2PBackendConfig(
        backend="mfa",
        language="de",
        source_language="de",
        model_path="/tmp/mfa_models/german_mfa.zip",
        mfa_command="/opt/conda/envs/g2p/bin/mfa",
    )

    metadata = g2p_metadata(config)

    assert metadata["g2p_model_path"] == "german_mfa"
    assert metadata["mfa_command"] is None


def test_no_gpl_g2p_references_in_acoustic_files() -> None:
    repo_root = Path(__file__).parents[4]
    production_files = [
        repo_root / "nemo_curator/stages/audio/text_filtering/acoustic_distractor.py",
        repo_root / "nemo_curator/stages/audio/text_filtering/g2p_backend.py",
        repo_root / "scripts/build_phoneme_vocab.py",
        repo_root / "examples/audio/text_processing/run_text_pipeline.py",
    ]
    for path in production_files:
        text = path.read_text(encoding="utf-8").lower()
        assert "phonemizer" not in text
        assert "espeak" not in text
