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

import asyncio
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import httpx
from loguru import logger
from openai import AsyncOpenAI, OpenAI

from nemo_curator.models.client.llm_client import AsyncLLMClient, ConversationFormatter, GenerationConfig, LLMClient

_MIN_PROMETHEUS_SAMPLE_FIELDS = 2


@dataclass(frozen=True)
class QueueBackpressureConfig:
    """Client-side admission control driven by a Prometheus queue metric.

    This mirrors the queue gate used by the optimized Gemma serving recipe:
    requests pause while the server reports too many waiting requests. Metrics
    reads are cached briefly so a large client pool does not hammer ``/metrics``.
    """

    max_waiting_requests: int
    poll_interval_seconds: float = 0.25
    stale_after_seconds: float = 2.0
    retry_after_seconds: float | None = None
    metric_name: str = "vllm:num_requests_waiting"
    fail_open: bool = True
    metrics_url: str | None = None

    def __post_init__(self) -> None:
        if self.max_waiting_requests < 1:
            msg = f"max_waiting_requests must be >= 1, got {self.max_waiting_requests}"
            raise ValueError(msg)
        if self.poll_interval_seconds <= 0:
            msg = f"poll_interval_seconds must be > 0, got {self.poll_interval_seconds}"
            raise ValueError(msg)
        if self.stale_after_seconds < 0:
            msg = f"stale_after_seconds must be >= 0, got {self.stale_after_seconds}"
            raise ValueError(msg)
        if self.retry_after_seconds is not None and self.retry_after_seconds <= 0:
            msg = f"retry_after_seconds must be > 0 when set, got {self.retry_after_seconds}"
            raise ValueError(msg)
        if not self.metric_name.strip():
            msg = "metric_name must not be empty"
            raise ValueError(msg)


def _default_metrics_url(base_url: str) -> str:
    """Derive the server metrics endpoint from an OpenAI ``.../v1`` URL."""
    root = base_url.rstrip("/")
    root = root.removesuffix("/v1")
    return f"{root}/metrics"


def _prometheus_metric_total(payload: str, metric_name: str) -> float:
    """Sum all labelled series for ``metric_name`` in Prometheus text."""
    values: list[float] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < _MIN_PROMETHEUS_SAMPLE_FIELDS:
            continue
        series_name = fields[0].split("{", 1)[0]
        if series_name != metric_name:
            continue
        try:
            values.append(float(fields[1]))
        except ValueError:
            continue
    if not values:
        msg = f"Metric {metric_name!r} was not found in the Prometheus payload"
        raise ValueError(msg)
    return sum(values)


class OpenAIClient(LLMClient):
    """
    A wrapper around OpenAI's Python client for querying models
    """

    def __init__(self, **kwargs) -> None:
        # Extract timeout if provided, default to 120 for backward compatibility
        self.timeout = kwargs.pop("timeout", 120)
        self.openai_kwargs = kwargs

    def setup(self) -> None:
        """
        Setup the client.
        """
        self.client = OpenAI(**self.openai_kwargs)

    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        if conversation_formatter is not None:
            warnings.warn("conversation_formatter is not used in an OpenAIClient", stacklevel=2)

        # Use default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)

        if generation_config.top_k is not None:
            warnings.warn("top_k is not used in an OpenAIClient", stacklevel=2)

        create_kwargs = {
            "messages": messages,
            "model": model,
            "max_tokens": generation_config.max_tokens,
            "n": generation_config.n,
            "seed": generation_config.seed,
            "stop": generation_config.stop,
            "stream": generation_config.stream,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "timeout": self.timeout,
        }
        if generation_config.extra_kwargs:
            overlapping = set(generation_config.extra_kwargs) & set(create_kwargs)
            if overlapping:
                logger.warning(f"extra_kwargs will overwrite existing parameter(s): {overlapping}")
            create_kwargs.update(generation_config.extra_kwargs)

        if not hasattr(self, "client"):
            self.setup()

        response = self.client.chat.completions.create(**create_kwargs)

        return [choice.message.content for choice in response.choices]


class AsyncOpenAIClient(AsyncLLMClient):
    """
    A wrapper around OpenAI's Python async client for querying models
    """

    def __init__(
        self,
        max_concurrent_requests: int = 5,
        max_retries: int = 3,
        base_delay: float = 1.0,
        queue_backpressure: QueueBackpressureConfig | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the AsyncOpenAI client.

        Args:
            max_concurrent_requests: Maximum number of concurrent requests
            max_retries: Maximum number of retry attempts for rate-limited requests
            base_delay: Base delay for exponential backoff (in seconds)
            queue_backpressure: Optional Prometheus-driven queue admission
                control. When configured, chat requests wait until the server's
                queue is below ``max_waiting_requests``.
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(max_concurrent_requests, max_retries, base_delay)
        # Extract timeout if provided, default to 120 for backward compatibility
        self.timeout = kwargs.pop("timeout", 120)
        self.queue_backpressure = queue_backpressure
        self._queue_metrics_url: str | None = None
        if queue_backpressure is not None:
            base_url = str(kwargs.get("base_url", ""))
            self._queue_metrics_url = queue_backpressure.metrics_url or (
                _default_metrics_url(base_url) if base_url else None
            )
            if not self._queue_metrics_url:
                msg = "queue_backpressure requires metrics_url or an OpenAI base_url"
                raise ValueError(msg)
        self._queue_metric_cache: tuple[float | None, float] | None = None
        self._queue_metric_lock: asyncio.Lock | None = None
        self._queue_metric_lock_loop: asyncio.AbstractEventLoop | None = None
        self.openai_kwargs = kwargs

    def setup(self) -> None:
        """
        Setup the client.
        """
        self.client = AsyncOpenAI(**self.openai_kwargs)

    async def _read_waiting_requests(self) -> float | None:
        """Read and briefly cache the configured server queue metric."""
        config = self.queue_backpressure
        if config is None or self._queue_metrics_url is None:
            return None

        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._queue_metric_cache is not None:
            cached_value, cached_at = self._queue_metric_cache
            if now - cached_at <= config.stale_after_seconds:
                return cached_value

        if self._queue_metric_lock is None or self._queue_metric_lock_loop is not loop:
            self._queue_metric_lock = asyncio.Lock()
            self._queue_metric_lock_loop = loop

        async with self._queue_metric_lock:
            now = loop.time()
            if self._queue_metric_cache is not None:
                cached_value, cached_at = self._queue_metric_cache
                if now - cached_at <= config.stale_after_seconds:
                    return cached_value

            try:
                async with httpx.AsyncClient(timeout=10.0) as metrics_client:
                    response = await metrics_client.get(self._queue_metrics_url)
                    response.raise_for_status()
                value = _prometheus_metric_total(response.text, config.metric_name)
            except (httpx.HTTPError, ValueError) as exc:
                if not config.fail_open:
                    raise
                # Cache a fail-open result too; otherwise every concurrent request
                # would immediately retry a metrics endpoint that is unavailable.
                self._queue_metric_cache = (None, now)
                logger.warning(
                    "Queue backpressure metrics unavailable at {} (fail_open=True): {}",
                    self._queue_metrics_url,
                    exc,
                )
                return None
            else:
                self._queue_metric_cache = (value, now)
                return value

    async def _wait_for_queue_capacity(self) -> None:
        """Pause admission while the server's waiting queue is at its cap."""
        config = self.queue_backpressure
        if config is None:
            return

        while True:
            waiting = await self._read_waiting_requests()
            if waiting is None or waiting < config.max_waiting_requests:
                return
            delay = config.retry_after_seconds or config.poll_interval_seconds
            logger.debug(
                "Server queue backpressure active: {}={} (limit={}); retrying in {:.2f}s",
                config.metric_name,
                waiting,
                config.max_waiting_requests,
                delay,
            )
            await asyncio.sleep(delay)

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        """
        Internal implementation of query_model without retry/concurrency logic.
        """
        if conversation_formatter is not None:
            warnings.warn("conversation_formatter is not used in an AsyncOpenAIClient", stacklevel=2)

        # Use default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)

        if generation_config.top_k is not None:
            warnings.warn("top_k is not used in an AsyncOpenAIClient", stacklevel=2)

        create_kwargs = {
            "messages": messages,
            "model": model,
            "max_tokens": generation_config.max_tokens,
            "n": generation_config.n,
            "seed": generation_config.seed,
            "stop": generation_config.stop,
            "stream": generation_config.stream,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "timeout": self.timeout,
        }
        if generation_config.extra_kwargs:
            overlapping = set(generation_config.extra_kwargs) & set(create_kwargs)
            if overlapping:
                logger.warning(f"extra_kwargs will overwrite existing parameter(s): {overlapping}")
            create_kwargs.update(generation_config.extra_kwargs)

        if not hasattr(self, "client"):
            self.setup()

        await self._wait_for_queue_capacity()

        response = await self.client.chat.completions.create(**create_kwargs)

        return [choice.message.content for choice in response.choices]
