"""Neutral core: events, provider interface, dispatch, bounded flush."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import pluggy

from conda_otel.events import matched_events

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from typing import Any

    from conda.base.context import Context
    from opentelemetry.sdk._logs.export import LogRecordExporter

    from conda_otel.events import EventSelector
    from conda_otel.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ProviderSpecs:
    """Extension point for conda plugins that supply telemetry providers."""

    @pluggy.HookspecMarker("conda")
    def conda_otel_providers(self, context: Context) -> Iterable[Provider]:
        """Return providers enabled by their own conda settings."""


@dataclass
class Event:
    """A vendor-neutral event.

    `attributes` holds the raw values from conda. conda-otel does not redact them,
    so each provider must redact them before it sends them. For install events the
    keys are: command, channels, channel_priority, requested_packages,
    resolved_packages (success), missing_packages and exception_name (not found).
    """

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Provider:
    """What a conda plugin returns from its provider hook.

    `name` identifies the provider in flush results and debug logs.
    `exporter` sends the log records. `resource_attributes` are set once on the
    provider's own OTel pipeline. `events` lists the `EventSelector` values the
    provider subscribes to. An empty list means it receives nothing.
    """

    name: str
    exporter: LogRecordExporter
    events: Iterable[EventSelector] = field(kw_only=True)
    resource_attributes: dict[str, Any] = field(default_factory=dict)
    #: Return a changed event, or None to drop it for this provider.
    transform: Callable[[Event], Event | None] | None = None


@contextmanager
def ignore_otel_environment() -> Iterator[None]:
    """Hide all `OTEL_*` variables while OTel objects are built.

    The SDK and the exporters read these variables when they are created.
    A user must not be able to redirect, disable or change the events with them.
    """
    # tuple() copies the names, so removing variables in the loop is safe.
    saved = {
        var: os.environ.pop(var) for var in tuple(os.environ) if var.startswith("OTEL_")
    }
    try:
        yield
    finally:
        os.environ.update(saved)


def is_scalar(value: Any) -> bool:
    """Return whether the value is a string, a boolean or a number."""
    return isinstance(value, (str, bool, int, float))


def clean_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Keep scalar values and lists of one scalar type. Drop all other values."""
    cleaned = {}
    for key, value in attributes.items():
        if is_scalar(value):
            cleaned[key] = value
        elif (
            isinstance(value, (list, tuple))
            and all(is_scalar(v) for v in value)
            and len({type(v) for v in value}) <= 1
        ):
            cleaned[key] = list(value)
    return cleaned


def load_providers(context: Context) -> list[Provider]:
    """Discover providers registered with conda's plugin manager.

    Note: pluggy propagates a provider hook failure, so no provider loads for
    that command.
    """
    manager = context.plugin_manager
    hook = getattr(manager.hook, "conda_otel_providers", None)
    if hook is None or hook.spec is None:
        manager.add_hookspecs(ProviderSpecs)
    with ignore_otel_environment():
        return [
            provider
            for group in manager.hook.conda_otel_providers(context=context)
            for provider in group
        ]


class Dispatcher:
    """Send each event to providers subscribed to it."""

    def __init__(self, providers: list[Provider]):
        """Start a pipeline for each provider."""
        # lazy import for performance
        from conda_otel.pipeline import Pipeline

        self.pipelines: list[Pipeline] = []
        for provider in providers:
            try:
                event_names = matched_events(provider.events)
                if not event_names:
                    continue
                self.pipelines.append(Pipeline(provider, event_names))
            except ValueError as exc:
                logger.warning(
                    "Skipping provider %s: invalid events have been declared (%s). "
                    "Use a list of EventSelector values from conda_otel.events.",
                    provider.name,
                    exc,
                )
            except Exception:
                logger.debug(
                    "Skipping provider %s: it failed to start due to an error.",
                    provider.name,
                    exc_info=True,
                )

    def __bool__(self) -> bool:
        """Return whether at least one provider pipeline started."""
        return bool(self.pipelines)

    def emit(self, event: Event) -> None:
        """Copy the event for each provider, transform it, clean it, and queue it.

        A provider whose transform returns None does not get the event.
        """
        for pipe in self.pipelines:
            if event.name not in pipe.events:
                continue
            copy = replace(event, attributes=dict(event.attributes))
            if pipe.provider.transform is not None:
                copy = pipe.provider.transform(copy)
                if copy is None:
                    continue
            pipe.emit(replace(copy, attributes=clean_attributes(copy.attributes)))

    def flush(self, budget: float = 3.0) -> dict[str, bool]:
        """Flush all providers in parallel and wait at most ``budget`` seconds.

        Return a map of provider name to whether its events were delivered.
        A provider with nothing to send counts as delivered.
        Note: a slow flush keeps running in the background. Its data is lost
        when the process exits.
        """
        results: dict[str, bool] = {}

        def run(pipe: Pipeline) -> None:
            """Flush one provider and save the result."""
            try:
                ok = pipe.logger_provider.force_flush(int(budget * 1000))
            except Exception:
                ok = False
            failed = pipe.exporter.delivered is False
            results[pipe.provider.name] = bool(ok) and not failed

        deadline = time.monotonic() + budget
        threads = [
            threading.Thread(target=run, args=(p,), daemon=True) for p in self.pipelines
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        return {
            p.provider.name: results.get(p.provider.name, False) for p in self.pipelines
        }
