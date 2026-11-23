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
from conftest import RAW_CHANNELS, REDACTED_CHANNELS, SECRET
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


# Synthetic test data. Every secret contains SECRET.
PACKAGE = "numpy"
ATTRS = {"command": "install", "channel_priority": "flexible"}
RECORD = "numpy=1.0=0"  # resolved package as name=version=build
RAW_ATTRIBUTES = {
    "command": "install",
    "channels": RAW_CHANNELS,
    "channel_priority": "flexible",
    "requested_packages": [PACKAGE],
    "resolved_packages": [RECORD],
    "missing_packages": [
        "https://user:SECRET@host.example/x-1.0-0.conda"
    ],  # URL, not a name
    "exception_name": "FooPluginError",  # not in the allowlist
    "undeclared": SECRET,
}
SANITIZED_ATTRIBUTES = {
    "command": "install",
    "channels": REDACTED_CHANNELS,
    "channel_priority": "flexible",
    "requested_packages": [PACKAGE],
    "resolved_packages": [RECORD],
}


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param("conda-forge", "conda-forge", id="name"),
        pytest.param("local", "local", id="name-local"),  # conda's local build channel
        pytest.param("conda-forge/label/dev", "conda-forge/label/dev", id="name-label"),
        pytest.param(
            "https://host.example/chan", "https://host.example/chan", id="url-plain"
        ),
        pytest.param(RAW_CHANNELS[0], REDACTED_CHANNELS[0], id="userinfo"),
        # conda's own token pattern misses "_" and ".".
        pytest.param(
            "https://host.example/t/SECRET_TO.KEN/chan",
            "https://host.example/chan",
            id="token",
        ),
        pytest.param(
            "https://host.example/prefix/t/SECRET/chan",
            "https://host.example/prefix/chan",
            id="token-after-prefix",
        ),
        pytest.param(
            "https://host.example/chan?token=SECRET#SECRET",
            "https://host.example/chan",
            id="query-fragment",
        ),
        pytest.param(
            "https://user:SECRET@host.example:1234/t/SECRET/chan/label/dev?x=SECRET",
            "https://host.example:1234/chan/label/dev",
            id="all-with-port",
        ),
        pytest.param(
            "HTTPS://user:SECRET@Host.example/chan",
            "https://Host.example/chan",
            id="upper-scheme",
        ),  # urlsplit lowercases the scheme
        pytest.param(
            "https://us@er:SECRET@host.example/chan",
            "https://host.example/chan",
            id="userinfo-two-at",
        ),  # split at the last @
        pytest.param(
            "http://user:SECRET@[::1]:1234/t/SECRET/chan",
            "http://[::1]:1234/chan",
            id="ipv6",
        ),
        pytest.param(
            "user:SECRET@host.example/t/SECRET/chan?x=SECRET",
            "host.example/chan",
            id="no-scheme",
        ),
        pytest.param(RAW_CHANNELS[3], "<local>", id="file-url"),
        pytest.param("FILE:///C:/SECRET", "<local>", id="file-url-upper"),
        pytest.param("/home/SECRET/chan", "<local>", id="posix"),
        pytest.param("/c/Users/SECRET/chan", "<local>", id="msys"),
        pytest.param("./SECRET", "<local>", id="relative"),
        pytest.param("~/SECRET", "<local>", id="home"),
        pytest.param(r"C:\Users\SECRET\chan", "<local>", id="windows-backslash"),
        pytest.param("C:/Users/SECRET/chan", "<local>", id="windows-slash"),
        pytest.param(r"\\server\SECRET\chan", "<local>", id="unc"),
        pytest.param("http://host.example:notaport/chan", "<invalid>", id="bad-port"),
        pytest.param("https://[::1/chan", "<invalid>", id="bad-ipv6"),
        pytest.param("https:///chan", "<invalid>", id="no-host"),
        # L1 and L2 both redact, so a second pass must change nothing.
        pytest.param("<local>", "<local>", id="marker-local"),
        pytest.param("<invalid>", "<invalid>", id="marker-invalid"),
    ],
)
def test_redact_channel(value: str, expected: str) -> None:
    """Secrets and local paths are removed. Plain channel names are kept."""
    result = core.redact_channel(value)
    assert result == expected
    assert type(result) is str


def test_emit_sanitizes_before_transform(
    in_memory: Provider, caplog: pytest.LogCaptureFixture
) -> None:
    """Providers get only declared, clean attributes, and the log names no value."""
    assert set(RAW_ATTRIBUTES) - {"undeclared"} == set(core.ATTRIBUTES)
    seen = []

    def transform(event: Event) -> Event:
        """Keep the provider input for the privacy check."""
        seen.append(event)
        return event

    in_memory.transform = transform
    dispatcher = Dispatcher([in_memory])
    with caplog.at_level(logging.DEBUG):
        dispatcher.emit(Event(EventName.SUCCESS, RAW_ATTRIBUTES))
        dispatcher.flush()
    assert [event.attributes for event in seen] == [SANITIZED_ATTRIBUTES]
    assert records(in_memory) == [
        {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in SANITIZED_ATTRIBUTES.items()
        }
    ]
    assert "undeclared" in caplog.text
    assert SECRET not in caplog.text


@pytest.mark.parametrize(
    "key, value",
    [
        pytest.param("command", "rm -rf /", id="unknown-command"),
        pytest.param("channel_priority", "bogus", id="unknown-priority"),
        pytest.param("exception_name", "FooPluginError", id="unknown-exception"),
        pytest.param("requested_packages", None, id="none"),
        pytest.param("requested_packages", [1234], id="not-a-string"),
        pytest.param("channels", [1234], id="channel-not-a-string"),
    ],
)
def test_bad_values_are_dropped(key: str, value: Any) -> None:
    """A value that fails its check is dropped and does not raise."""
    assert key not in core.sanitize_attributes({key: value})


def records(provider: Provider) -> list[dict[str, Any]]:
    """Return the attributes of each record the provider exported."""
    return [
        dict(r.log_record.attributes) for r in provider.exporter.get_finished_logs()
    ]


def test_emit_adds_event_name_and_drops_non_scalar_values() -> None:
    """The event name is added. Nested and mixed-type values are dropped."""
    # clean_attributes runs after the transform, so these keys reach it.
    provider = Provider(
        "p",
        InMemoryLogRecordExporter(),
        events=SUCCESS_ONLY,
        transform=lambda e: Event(
            e.name, {"a": 1, "nested": {"x": 1}, "mixed": [1, "a"], "ok": ["a"]}
        ),
    )
    dispatcher = Dispatcher([provider])
    dispatcher.emit(Event(EventName.SUCCESS))
    dispatcher.flush()
    assert records(provider) == [{"a": 1, "ok": ("a",)}]
    (exported,) = provider.exporter.get_finished_logs()
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
    dropping = Provider(
        "dropping",
        InMemoryLogRecordExporter(),
        events=SUCCESS_ONLY,
        transform=lambda e: None,
    )
    changing = Provider(
        "changing",
        InMemoryLogRecordExporter(),
        events=SUCCESS_ONLY,
        transform=lambda e: Event("renamed", {"k": "v"}),
    )
    event = Event(EventName.SUCCESS, ATTRS)
    dispatcher = Dispatcher([in_memory, dropping, changing])
    dispatcher.emit(event)
    dispatcher.flush()
    assert records(in_memory) == [ATTRS]
    assert records(dropping) == []
    assert records(changing) == [{"k": "v"}]
    assert event.attributes == ATTRS


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
    dispatcher.emit(Event(EventName.SUCCESS, ATTRS))
    dispatcher.flush()
    (record,) = in_memory.exporter.get_finished_logs()
    assert dict(record.log_record.attributes) == ATTRS
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
    dummy_data = ATTRS
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
        InMemoryLogRecordExporter(),
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
        "d.emit(Event(EventName.SUCCESS, "
        "{'command': 'install', 'channel_priority': 'flexible'}))\n"
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
    assert out.strip() == "[('channel_priority', 'flexible'), ('command', 'install')]"


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
    dummy_data = ATTRS
    dispatcher = Dispatcher([provider])
    dispatcher.emit(Event(EventName.SUCCESS, dummy_data))
    dispatcher.flush()
    assert records(provider) == [dummy_data]


@pytest.mark.parametrize(
    ("subscription", "event"),
    [
        pytest.param(SUCCESS_ONLY, EventName.PACKAGES_NOT_FOUND, id="not-subscribed"),
        pytest.param([ALL_EVENTS], "example.success", id="unsupported"),
    ],
)
def test_unmatched_event_skips_provider(
    subscription: list[EventSelector], event: str
) -> None:
    """An event outside the subscription skips the transform and the export."""
    calls: list[Event] = []

    def transform(event: Event) -> Event:
        calls.append(event)
        return event

    provider = Provider(
        "p", InMemoryLogRecordExporter(), events=subscription, transform=transform
    )
    dispatcher = Dispatcher([provider])
    dispatcher.emit(Event(event))
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
    dummy_data = ATTRS
    dispatcher.emit(Event(EventName.SUCCESS, dummy_data))
    assert dispatcher.flush() == {"in-memory": True}
    assert records(in_memory) == [dummy_data]


@pytest.mark.parametrize(
    ("subscription", "empty_catalog"),
    [
        pytest.param([], False, id="empty-declaration"),
        pytest.param([ALL_EVENTS], True, id="wildcard-no-matches"),
    ],
)
def test_no_matching_events_skips_pipeline(
    subscription: list[EventSelector],
    empty_catalog: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Start no pipeline and log nothing when the subscription matches no events."""
    if empty_catalog:
        monkeypatch.setattr(events, "SUPPORTED_COMMAND_OUTCOME_PAIRS", frozenset())
    provider = Provider("p", InMemoryLogRecordExporter(), events=subscription)
    with caplog.at_level(logging.DEBUG):
        dispatcher = Dispatcher([provider])
    assert dispatcher.pipelines == []
    assert caplog.records == []
