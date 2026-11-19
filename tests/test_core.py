from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pluggy
import pytest
from conda.plugins import hookimpl
from conda.plugins.manager import CondaPluginManager
from opentelemetry import _logs
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    LogRecordExporter,
    LogRecordExportResult,
)

from conda_otel import core, events
from conda_otel.core import Dispatcher, Event, Provider
from conda_otel.events import (
    ALL_EVENTS,
    Command,
    EventName,
    EventSelector,
    Outcome,
)

if TYPE_CHECKING:
    from typing import Any


SUCCESS_ONLY = [EventSelector.exact(Command.INSTALL, Outcome.SUCCESS)]


def records(provider: Provider) -> list[dict[str, Any]]:
    """Return the attributes of each record the provider exported."""
    return [
        dict(r.log_record.attributes) for r in provider.exporter.get_finished_logs()
    ]


def test_emit_adds_event_name_and_drops_non_scalar_values(in_memory: Provider) -> None:
    """The event name is added. Nested and mixed-type values are dropped."""
    dispatcher = Dispatcher([in_memory])
    dispatcher.emit(
        Event(
            EventName.SUCCESS,
            {"a": 1, "nested": {"x": 1}, "mixed": [1, "a"], "ok": ["a"]},
        )
    )
    dispatcher.flush()
    assert records(in_memory) == [{"a": 1, "ok": ("a",)}]
    (exported,) = in_memory.exporter.get_finished_logs()
    assert exported.log_record.event_name == EventName.SUCCESS


def test_no_providers_is_falsy_and_does_nothing() -> None:
    """With no provider, the dispatcher is false and sends nothing."""
    dispatcher = Dispatcher([])
    assert not dispatcher
    dispatcher.emit(Event(EventName.SUCCESS))
    assert dispatcher.flush() == {}


def test_global_logger_provider_is_not_used(in_memory: Provider) -> None:
    """The dispatcher does not set the global OTel logger provider."""
    dispatcher = Dispatcher([in_memory])
    dispatcher.emit(Event(EventName.SUCCESS))
    dispatcher.flush()
    assert not isinstance(_logs.get_logger_provider(), LoggerProvider)


def test_transform_is_per_provider_and_can_drop(in_memory: Provider) -> None:
    """Each provider changes or drops the event on its own copy."""
    # Get the class of the fixture's exporter.
    exporter_class = in_memory.exporter.__class__
    dropping = Provider(
        "dropping", exporter_class(), events=SUCCESS_ONLY, transform=lambda e: None
    )
    changing = Provider(
        "changing",
        exporter_class(),
        events=SUCCESS_ONLY,
        transform=lambda e: Event("renamed", {"k": "v"}),
    )
    event = Event(EventName.SUCCESS, {"secret": "x"})
    dispatcher = Dispatcher([in_memory, dropping, changing])
    dispatcher.emit(event)
    dispatcher.flush()
    assert records(in_memory) == [{"secret": "x"}]
    assert records(dropping) == []
    assert records(changing) == [{"k": "v"}]
    assert event.attributes == {"secret": "x"}


@pytest.mark.parametrize(
    "var, value",
    [
        ("OTEL_SDK_DISABLED", "true"),
        ("OTEL_RESOURCE_ATTRIBUTES", "injected=yes"),
    ],
)
@pytest.mark.parametrize("when", ["before_build", "before_emit"])
def test_otel_environment_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    in_memory: Provider,
    var: str,
    value: str,
    when: str,
) -> None:
    """OTEL_* variables do not disable the SDK or add resource attributes."""
    in_memory.resource_attributes = {"service.name": "svc"}
    if when == "before_build":
        monkeypatch.setenv(var, value)
    dispatcher = Dispatcher([in_memory])
    if when == "before_emit":
        monkeypatch.setenv(var, value)
    dispatcher.emit(Event(EventName.SUCCESS, {"a": "abcdefgh", "b": 1, "c": 2}))
    dispatcher.flush()
    (record,) = in_memory.exporter.get_finished_logs()
    assert dict(record.log_record.attributes) == {
        "a": "abcdefgh",
        "b": 1,
        "c": 2,
    }
    assert "injected" not in record.resource.attributes


def test_otel_variables_are_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The variables come back after the context ends."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    with core.ignore_otel_environment():
        assert "OTEL_SDK_DISABLED" not in os.environ
    assert os.environ["OTEL_SDK_DISABLED"] == "true"


def make_context(*plugins: object) -> SimpleNamespace:
    """Return a fake conda context whose plugin manager has the plugins."""
    manager = CondaPluginManager()
    for plugin in plugins:
        manager.register(plugin)
    return SimpleNamespace(plugin_manager=manager)


class ValidPlugin:
    """Plugin that supplies one working provider."""

    def __init__(self, provider: Provider) -> None:
        """Keep the provider that the hook returns."""
        self.provider = provider

    @hookimpl(optionalhook=True)
    def conda_otel_providers(self, context):
        """Yield the provider."""
        yield self.provider


class FailsAtCallPlugin:
    """Plugin whose hook raises when conda calls it."""

    @hookimpl(optionalhook=True)
    def conda_otel_providers(self, context):
        """Raise before any provider is returned."""
        raise RuntimeError("provider failed")


class FailsAfterYieldPlugin:
    """Plugin whose hook yields one provider and then raises."""

    @hookimpl(optionalhook=True)
    def conda_otel_providers(self, context):
        """Yield a provider, then raise while conda reads the result."""
        yield Provider("partial", InMemoryLogRecordExporter(), events=SUCCESS_ONLY)
        raise RuntimeError("provider failed")


def test_conda_plugin_provider_is_discovered(in_memory: Provider) -> None:
    """An ordinary conda plugin supplies a provider through the core hook."""
    context = make_context(ValidPlugin(in_memory))
    dummy_data = {"a": 1}
    dispatcher = Dispatcher(core.load_providers(context))
    dispatcher.emit(Event(EventName.SUCCESS, dummy_data))
    assert dispatcher.flush() == {"in-memory": True}
    assert records(in_memory) == [dummy_data]
    # A second call must not register the hookspec again.
    assert core.load_providers(context) == [in_memory]


def test_no_provider_hook_returns_nothing() -> None:
    """With no plugin that implements the hook, discovery finds no provider."""
    assert core.load_providers(make_context()) == []


def test_providers_are_discovered_without_otel_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hooks run without OTEL_* variables. The variables are restored after."""
    seen: dict[str, str] = {}

    class EnvPlugin:
        @hookimpl(optionalhook=True)
        def conda_otel_providers(self, context):
            seen.update({k: v for k, v in os.environ.items() if k.startswith("OTEL_")})
            return []

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env.invalid:4318")
    assert core.load_providers(make_context(EnvPlugin())) == []
    assert seen == {}
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://env.invalid:4318"


def test_hookspec_rejects_implementation_with_wrong_argument_name() -> None:
    """The hookspec checks the signature of each implementation."""

    class WrongArgumentPlugin:
        @hookimpl(optionalhook=True)
        def conda_otel_providers(self, ctx):
            yield from ()

    with pytest.raises(pluggy.PluginValidationError, match="ctx"):
        core.load_providers(make_context(WrongArgumentPlugin()))


@pytest.mark.parametrize("bad_plugin", [FailsAtCallPlugin, FailsAfterYieldPlugin])
def test_failing_provider_aborts_discovery(
    in_memory: Provider, bad_plugin: type
) -> None:
    """A failing hook raises out of discovery. The other providers are lost."""
    context = make_context(ValidPlugin(in_memory), bad_plugin())
    with pytest.raises(RuntimeError, match="provider failed"):
        core.load_providers(context)


def test_provider_that_fails_to_start_is_skipped(in_memory: Provider) -> None:
    """A provider whose pipeline cannot start is left out. The others still work."""
    broken = Provider(
        "broken",
        in_memory.exporter.__class__(),
        events=SUCCESS_ONLY,
        resource_attributes=1,
    )
    dispatcher = Dispatcher([broken, in_memory])
    assert [p.provider.name for p in dispatcher.pipelines] == ["in-memory"]


class SlowExporter(LogRecordExporter):
    """Exporter that waits before it reports success."""

    def __init__(self, delay: float) -> None:
        """Set how many seconds each export waits."""
        self.delay = delay

    def export(self, batch: Any) -> LogRecordExportResult:
        """Wait, then report success."""
        time.sleep(self.delay)
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        """Do nothing."""


def test_flush_is_bounded_and_isolated(in_memory: Provider) -> None:
    """A slow provider does not block a fast one, and flush ends within the budget."""
    dispatcher = Dispatcher(
        [Provider("slow", SlowExporter(3), events=SUCCESS_ONLY), in_memory]
    )
    dispatcher.emit(Event(EventName.SUCCESS))
    start = time.monotonic()
    result = dispatcher.flush(budget=0.5)
    assert time.monotonic() - start < 1.5
    assert result == {"slow": False, "in-memory": True}
    assert len(records(in_memory)) == 1


def test_attribute_limit_variables_set_before_start_are_ignored() -> None:
    """The SDK reads `OTEL_ATTRIBUTE_*` at import, so test in a new process."""
    code = (
        "from conda_otel.core import Dispatcher, Event, Provider\n"
        "from conda_otel.events import Command, EventName, EventSelector, Outcome\n"
        "from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter\n"
        "sel = EventSelector.exact(Command.INSTALL, Outcome.SUCCESS)\n"
        "p = Provider('m', InMemoryLogRecordExporter(), events=[sel])\n"
        "d = Dispatcher([p])\n"
        "d.emit(Event(EventName.SUCCESS, {'a': 'abcdefgh', 'b': 1, 'c': 2}))\n"
        "d.flush()\n"
        "(r,) = p.exporter.get_finished_logs()\n"
        "print(sorted(r.log_record.attributes.items()))\n"
    )
    env = {
        **os.environ,
        "OTEL_ATTRIBUTE_COUNT_LIMIT": "1",
        "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT": "3",
        "PYTHONPATH": str(Path(core.__file__).parent.parent),
    }
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.strip() == "[('a', 'abcdefgh'), ('b', 1), ('c', 2)]"


def test_otel_log_records_do_not_reach_the_root_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Conda prints root logger records to stderr, so OTel records must not reach it."""
    import conda_otel.pipeline  # noqa: F401  (sets up the OTel logger)

    logging.getLogger("opentelemetry.sdk._logs.export").warning("export failed")
    assert "export failed" not in caplog.text


def test_importing_hooks_does_not_import_the_otel_sdk() -> None:
    """The plugin loads on every conda command, so the SDK must load lazily."""
    code = (
        "import sys\n"
        "import conda_otel.hooks\n"
        "print('opentelemetry.sdk' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(Path(core.__file__).parent.parent)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.strip() == "False"


def test_events_are_required() -> None:
    """A provider without `events` cannot be built."""
    with pytest.raises(TypeError, match="events"):
        Provider("p", InMemoryLogRecordExporter())  # type: ignore[call-arg]


def test_matching_event_is_exported() -> None:
    """An event that the provider subscribed to is exported."""
    provider = Provider("p", InMemoryLogRecordExporter(), events=SUCCESS_ONLY)
    dummy_data = {"a": 1}
    dispatcher = Dispatcher([provider])
    dispatcher.emit(Event(EventName.SUCCESS, dummy_data))
    dispatcher.flush()
    assert records(provider) == [dummy_data]


def test_non_matching_event_never_reaches_the_provider() -> None:
    """An event outside the subscription skips the transform and the export."""
    calls: list[Event] = []

    def transform(event: Event) -> Event:
        calls.append(event)
        return event

    provider = Provider(
        "p", InMemoryLogRecordExporter(), events=SUCCESS_ONLY, transform=transform
    )
    dispatcher = Dispatcher([provider])
    dispatcher.emit(Event(EventName.PACKAGES_NOT_FOUND))
    dispatcher.flush()
    assert calls == []
    assert records(provider) == []


def test_transform_can_rename_the_event() -> None:
    """The exported name is the one the transform returns."""
    provider = Provider(
        "p",
        InMemoryLogRecordExporter(),
        events=SUCCESS_ONLY,
        transform=lambda e: Event("renamed"),
    )
    dispatcher = Dispatcher([provider])
    dispatcher.emit(Event(EventName.SUCCESS))
    dispatcher.flush()
    (exported,) = provider.exporter.get_finished_logs()
    assert exported.log_record.event_name == "renamed"


def test_all_events_does_not_match_unregistered_events() -> None:
    """An event that is not in the registry never reaches the transform."""
    calls: list[Event] = []

    def transform(event: Event) -> Event:
        calls.append(event)
        return event

    provider = Provider(
        "p", InMemoryLogRecordExporter(), events=[ALL_EVENTS], transform=transform
    )
    dispatcher = Dispatcher([provider])
    dispatcher.emit(Event("create.success"))
    dispatcher.flush()
    assert calls == []
    assert records(provider) == []


@pytest.mark.parametrize(
    "bad_events",
    [
        "",
        [ALL_EVENTS, "x"],
        [EventSelector.exact(Command.INSTALL, "foobar")],
        [EventSelector.for_command(None)],
    ],
)
def test_invalid_declaration_skips_only_that_provider(
    caplog: pytest.LogCaptureFixture, in_memory: Provider, bad_events: Any
) -> None:
    """An invalid declaration logs one warning. The other providers still work."""
    bad = Provider("bad", InMemoryLogRecordExporter(), events=bad_events)
    context = make_context(ValidPlugin(bad), ValidPlugin(in_memory))
    with caplog.at_level(logging.WARNING):
        dispatcher = Dispatcher(core.load_providers(context))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "bad" in warnings[0].getMessage()
    assert [p.provider.name for p in dispatcher.pipelines] == ["in-memory"]
    dummy_data = {"a": 1}
    dispatcher.emit(Event(EventName.SUCCESS, dummy_data))
    assert dispatcher.flush() == {"in-memory": True}
    assert records(in_memory) == [dummy_data]


def test_empty_declaration_starts_no_pipeline_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty list means the provider wants no events."""
    provider = Provider("p", InMemoryLogRecordExporter(), events=[])
    with caplog.at_level(logging.DEBUG):
        dispatcher = Dispatcher([provider])
    assert dispatcher.pipelines == []
    assert caplog.records == []


def test_wildcard_with_no_matches_starts_no_pipeline_and_logs_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A valid wildcard that matches nothing is not an error."""
    monkeypatch.setattr(events, "SUPPORTED_COMMAND_OUTCOME_PAIRS", frozenset())
    provider = Provider("p", InMemoryLogRecordExporter(), events=[ALL_EVENTS])
    with caplog.at_level(logging.DEBUG):
        dispatcher = Dispatcher([provider])
    assert dispatcher.pipelines == []
    assert caplog.records == []
