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

"""HTTP 429 queue admission and shared AIMD control for Dynamo/vLLM."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http
import math
import time
from dataclasses import dataclass

from aiohttp import ClientSession, ClientTimeout, web

_METRIC_NAME = "vllm:num_requests_waiting"
_POLL_INTERVAL_SECONDS = 0.25
_STALE_AFTER_SECONDS = 2.0
_RETRY_AFTER_SECONDS = 2.0
_REDUCE_FACTOR = 0.75
_ADDITIVE_INCREASE = 1
_SUCCESS_WINDOW = 25
_CEILING_OVERSHOOT = 0.10
_RAMPUP_SECONDS = 10.0
_CAPACITY_POLL_INTERVAL = 0.05
_MIN_PROMETHEUS_SAMPLE_FIELDS = 2
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
_EXEMPT_PATH_PREFIXES = (
    "/health",
    "/ready",
    "/metrics",
    "/version",
    "/v1/models",
    "/ping",
    "/is_scaling_elastic_ep",
)


def _prometheus_metric_total(payload: str, metric_name: str = _METRIC_NAME) -> float:
    """Sum all labelled series for one Prometheus metric."""
    values: list[float] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < _MIN_PROMETHEUS_SAMPLE_FIELDS or fields[0].split("{", 1)[0] != metric_name:
            continue
        with contextlib.suppress(ValueError):
            values.append(float(fields[1]))
    if not values:
        msg = f"Metric {metric_name!r} was not found"
        raise ValueError(msg)
    return sum(values)


@dataclass
class AIMDState:
    """Shared additive-increase/multiplicative-decrease concurrency window."""

    maximum: int
    success_window: int = _SUCCESS_WINDOW
    rampup_seconds: float = _RAMPUP_SECONDS

    def __post_init__(self) -> None:
        self.current_limit = 1 if self.rampup_seconds > 0 and self.maximum > 1 else self.maximum
        self.in_flight = 0
        self.blocked_until = 0.0
        self.success_streak = 0
        self.rate_limit_ceiling = 0
        self.consecutive_429s = 0
        self.rate_limit_events_total = 0
        self.multiplicative_decreases_total = 0
        self.additive_increases_total = 0
        self.rampup_started_at = time.monotonic()
        self.rampup_active = self.rampup_seconds > 0 and self.maximum > 1
        self._condition = asyncio.Condition()

    def _apply_ramp(self, now: float) -> None:
        if not self.rampup_active:
            return
        elapsed = max(0.0, now - self.rampup_started_at)
        if elapsed >= self.rampup_seconds:
            self.current_limit = self.maximum
            self.rampup_active = False
            return
        slots = math.floor((self.maximum - 1) * elapsed / self.rampup_seconds)
        self.current_limit = min(self.maximum, 1 + slots)

    def _soft_ceiling(self) -> int:
        if self.rate_limit_ceiling <= 0:
            return self.maximum
        overshoot = max(1, math.floor(self.rate_limit_ceiling * _CEILING_OVERSHOOT))
        return min(self.maximum, self.rate_limit_ceiling + overshoot)

    async def acquire(self) -> None:
        async with self._condition:
            while True:
                now = time.monotonic()
                self._apply_ramp(now)
                if now < self.blocked_until:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._condition.wait(), timeout=self.blocked_until - now)
                elif self.in_flight < self.current_limit:
                    self.in_flight += 1
                    return
                else:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._condition.wait(), timeout=_CAPACITY_POLL_INTERVAL)

    async def release_success(self) -> None:
        async with self._condition:
            self.in_flight = max(0, self.in_flight - 1)
            self.consecutive_429s = 0
            self._apply_ramp(time.monotonic())
            if self.rampup_active:
                self.success_streak = 0
            else:
                self.success_streak += 1
                if self.success_streak >= self.success_window:
                    previous = self.current_limit
                    self.current_limit = min(
                        self.current_limit + _ADDITIVE_INCREASE,
                        self._soft_ceiling(),
                    )
                    self.additive_increases_total += int(self.current_limit > previous)
                    self.success_streak = 0
            self._condition.notify_all()

    async def release_failure(self) -> None:
        async with self._condition:
            self.in_flight = max(0, self.in_flight - 1)
            if self.in_flight == 0:
                self.consecutive_429s = 0
            self._condition.notify_all()

    async def rate_limited(self, *, release_permit: bool) -> None:
        async with self._condition:
            if release_permit:
                self.in_flight = max(0, self.in_flight - 1)
            self.rampup_active = False
            previous = self.current_limit
            first_in_cascade = self.consecutive_429s == 0
            self.consecutive_429s += 1
            self.rate_limit_events_total += 1
            self.blocked_until = time.monotonic() + _RETRY_AFTER_SECONDS
            self.success_streak = 0
            if first_in_cascade:
                self.current_limit = max(1, math.floor(previous * _REDUCE_FACTOR))
                self.multiplicative_decreases_total += int(self.current_limit < previous)
                self.rate_limit_ceiling = (
                    previous if self.rate_limit_ceiling == 0 else min(self.rate_limit_ceiling, previous)
                )
            self._condition.notify_all()


class QueueMetricSampler:
    """Cache the least-loaded queue across automatically discovered workers."""

    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.value: float | None = None
        self.sampled_at = 0.0
        self._task: asyncio.Task | None = None

    def start(self, session: ClientSession) -> None:
        self._task = asyncio.create_task(self._run(session))

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    @staticmethod
    async def _read(session: ClientSession, url: str) -> float:
        async with session.get(url) as response:
            response.raise_for_status()
            return _prometheus_metric_total(await response.text())

    async def _run(self, session: ClientSession) -> None:
        while True:
            results = await asyncio.gather(
                *(self._read(session, url) for url in self.urls),
                return_exceptions=True,
            )
            values = [value for value in results if isinstance(value, float)]
            if values:
                # Big Iron lets its load balancer retry a 429 on another vLLM
                # backend. This centralized equivalent admits while any Dynamo
                # worker still has queue capacity; the frontend then routes it.
                self.value = min(values)
                self.sampled_at = time.monotonic()
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    def snapshot(self) -> tuple[float | None, bool]:
        fresh = self.value is not None and time.monotonic() - self.sampled_at <= _STALE_AFTER_SECONDS
        return (self.value if fresh else None), fresh


class AdmissionProxy:
    """OpenAI-compatible gateway that combines queue 429s with shared AIMD."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session: ClientSession | None = None
        self.sampler = QueueMetricSampler(args.metrics_url)
        self.aimd = AIMDState(maximum=args.max_concurrent_requests)
        self.forwarded_requests_total = 0
        self.queue_rejections_total = 0
        self.upstream_429_total = 0

    async def start(self, _app: web.Application) -> None:
        self.session = ClientSession(timeout=ClientTimeout(total=None), auto_decompress=False)
        self.sampler.start(self.session)

    async def close(self, _app: web.Application) -> None:
        await self.sampler.close()
        if self.session is not None:
            await self.session.close()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/metrics":
            return self._metrics()

        exempt = any(request.path.startswith(prefix) for prefix in _EXEMPT_PATH_PREFIXES)
        queue_depth, fresh = self.sampler.snapshot()
        if not exempt and fresh and queue_depth > self.args.max_waiting_requests:
            await self.aimd.rate_limited(release_permit=False)
            self.queue_rejections_total += 1
            return web.json_response(
                {
                    "error": {
                        "message": "vLLM waiting queue exceeded the admission threshold",
                        "type": "rate_limit_error",
                        "code": "queue_overloaded",
                        "queue_depth": queue_depth,
                        "max_waiting_requests": self.args.max_waiting_requests,
                    }
                },
                status=http.HTTPStatus.TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": str(int(_RETRY_AFTER_SECONDS)),
                    "X-Should-Retry": "true",
                    "X-Curator-AIMD-Limit": str(self.aimd.current_limit),
                },
            )

        permit_acquired = False
        if not exempt:
            await self.aimd.acquire()
            permit_acquired = True

        assert self.session is not None  # noqa: S101
        upstream_url = f"{self.args.upstream.rstrip('/')}{request.rel_url}"
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        try:
            self.forwarded_requests_total += 1
            async with self.session.request(
                request.method,
                upstream_url,
                headers=headers,
                data=request.content.iter_chunked(64 * 1024),
                allow_redirects=False,
            ) as upstream:
                response_headers = {
                    key: value for key, value in upstream.headers.items() if key.lower() not in _HOP_BY_HOP_HEADERS
                }
                response = web.StreamResponse(status=upstream.status, headers=response_headers)
                await response.prepare(request)
                async for chunk in upstream.content.iter_chunked(64 * 1024):
                    await response.write(chunk)
                await response.write_eof()

                if permit_acquired:
                    if upstream.status == http.HTTPStatus.TOO_MANY_REQUESTS:
                        self.upstream_429_total += 1
                        await self.aimd.rate_limited(release_permit=True)
                    elif upstream.status < http.HTTPStatus.BAD_REQUEST:
                        await self.aimd.release_success()
                    else:
                        await self.aimd.release_failure()
                return response
        except Exception:
            if permit_acquired:
                await self.aimd.release_failure()
            raise

    def _metrics(self) -> web.Response:
        queue_depth, fresh = self.sampler.snapshot()
        values = {
            "current_limit": self.aimd.current_limit,
            "in_flight": self.aimd.in_flight,
            "queue_metric_fresh": int(fresh),
            "forwarded_requests_total": self.forwarded_requests_total,
            "queue_rejections_total": self.queue_rejections_total,
            "upstream_429_total": self.upstream_429_total,
            "rate_limit_events_total": self.aimd.rate_limit_events_total,
            "multiplicative_decreases_total": self.aimd.multiplicative_decreases_total,
            "additive_increases_total": self.aimd.additive_increases_total,
        }
        if queue_depth is not None:
            values["worker_queue_depth"] = queue_depth
        lines = [f"curator_admission_{name} {value}" for name, value in values.items()]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - cluster-facing service
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--metrics-url", action="append", required=True)
    parser.add_argument("--max-waiting-requests", type=int, required=True)
    parser.add_argument("--max-concurrent-requests", type=int, default=8192)
    return parser


def main() -> None:
    args = _parser().parse_args()
    proxy = AdmissionProxy(args)
    app = web.Application(client_max_size=16 * 1024**2)
    app.router.add_route("*", "/{path_info:.*}", proxy.handle)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.close)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
