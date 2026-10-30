from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from conda_otel import core
from conda_otel.core import Dispatcher, Event, Provider
from opentelemetry import _logs
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult

if TYPE_CHECKING:
    from conftest import RecordingServer


def records(provider: Provider) -> list[dict[str, Any]]:
    """Return the attributes of each record the provider exported."""
    return [
        dict(r.log_record.attributes) for r in provider.exporter.get_finished_logs()
    ]


def test_emit_adds_event_name_and_drops_non_scalar_values(in_memory: Provider) -> None:
    """The event name is added. Nested and mixed-type values are dropped."""
    dispatcher = Dispatcher([in_memory])
    dispatcher.emit(
        Event("e", {"a": 1, "nested": {"x": 1}, "mixed": [1, "a"], "ok": ["a"]})
    )
    dispatcher.flush()
    assert records(in_memory) == [{"a": 1, "ok": ("a",)}]
    (exported,) = in_memory.exporter.get_finished_logs()
    assert exported.log_record.event_name == "e"


def test_no_providers_is_falsy_and_does_nothing() -> None:
    """With no provider, the dispatcher is false and sends nothing."""
    dispatcher = Dispatcher([])
    assert not dispatcher
    dispatcher.emit(Event("e"))
    assert dispatcher.flush() == {}


def test_global_logger_provider_is_not_used(in_memory: Provider) -> None:
    """The dispatcher does not set the global OTel logger provider."""
    dispatcher = Dispatcher([in_memory])
    dispatcher.emit(Event("e"))
    dispatcher.flush()
    assert not isinstance(_logs.get_logger_provider(), LoggerProvider)


def test_transform_is_per_provider_and_can_drop(in_memory: Provider) -> None:
    """Each provider changes or drops the event on its own copy."""
    exporter_class = in_memory.exporter.__class__
    dropping = Provider("dropping", exporter_class(), transform=lambda e: None)
    changing = Provider(
        "changing",
        exporter_class(),
        transform=lambda e: Event("renamed", {"k": "v"}),
    )
    event = Event("e", {"secret": "x"})
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
    dispatcher.emit(Event("e", {"a": "abcdefgh", "b": 1, "c": 2}))
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


def install_entry_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: str, source: str
) -> None:
    """Make a fake package that registers the Python code in `source` as a provider."""
    dist = tmp_path / f"{module}-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Name: {module}\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text(
        f"[{core.ENTRY_POINT_GROUP}]\n{module} = {module}:make\n"
    )
    (tmp_path / f"{module}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))


def otlp_provider_source(exporter_arguments: str = "") -> str:
    """Return the source of a provider that uses the OTLP HTTP exporter."""
    return (
        "from conda_otel.core import Provider\n"
        "from opentelemetry.exporter.otlp.proto.http._log_exporter import "
        "OTLPLogExporter\n"
        "def make(context):\n"
        f"    return Provider('fake', OTLPLogExporter({exporter_arguments}))\n"
    )


def test_exporter_is_built_without_otel_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An exporter with no endpoint must not use `OTEL_EXPORTER_OTLP_ENDPOINT`."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://env-endpoint.invalid:4318"
    )
    install_entry_point(monkeypatch, tmp_path, "fake_env", otlp_provider_source())
    (provider,) = core.load_providers(object())
    assert provider.exporter._endpoint == "http://localhost:4318/v1/logs"


def test_discovery_through_entry_point_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, otlp_server: RecordingServer
) -> None:
    """A registered provider is found and receives events."""
    source = otlp_provider_source(f"endpoint='{otlp_server.endpoint}/v1/logs'")
    install_entry_point(monkeypatch, tmp_path, "fake_found", source)
    dispatcher = Dispatcher(core.load_providers(object()))
    dispatcher.emit(Event("e", {"a": 1}))
    assert dispatcher.flush() == {"fake": True}
    ((path, _, body),) = otlp_server.received
    assert path == "/v1/logs"
    request = ExportLogsServiceRequest()
    request.ParseFromString(body)
    (record,) = request.resource_logs[0].scope_logs[0].log_records
    assert {kv.key for kv in record.attributes} == {"a"}
    assert record.event_name == "e"


def test_provider_that_fails_to_load_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One broken provider must not stop the other providers from loading."""
    install_entry_point(
        monkeypatch,
        tmp_path,
        "fake_broken",
        "def make(context):\n    raise RuntimeError('boom')\n",
    )
    install_entry_point(monkeypatch, tmp_path, "fake_good", otlp_provider_source())
    assert [p.name for p in core.load_providers(object())] == ["fake"]


def test_provider_that_fails_to_start_is_skipped(in_memory: Provider) -> None:
    """A provider whose pipeline cannot start is left out. The others still work."""
    broken = Provider("broken", in_memory.exporter.__class__(), resource_attributes=1)
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
    dispatcher = Dispatcher([Provider("slow", SlowExporter(3)), in_memory])
    dispatcher.emit(Event("e"))
    start = time.monotonic()
    result = dispatcher.flush(budget=0.5)
    assert time.monotonic() - start < 1.5
    assert result == {"slow": False, "in-memory": True}
    assert len(records(in_memory)) == 1


def test_attribute_limit_variables_set_before_start_are_ignored() -> None:
    """The SDK reads `OTEL_ATTRIBUTE_*` at import, so test in a new process."""
    code = (
        "from conda_otel.core import Dispatcher, Event, Provider\n"
        "from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter\n"
        "p = Provider('m', InMemoryLogRecordExporter())\n"
        "d = Dispatcher([p])\n"
        "d.emit(Event('e', {'a': 'abcdefgh', 'b': 1, 'c': 2}))\n"
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
    assert (
        out.strip()
        == "[('a', 'abcdefgh'), ('b', 1), ('c', 2)]"
    )


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
