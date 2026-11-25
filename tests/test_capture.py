from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from conda.base.context import Context
from conda.exceptions import PackagesNotFoundInChannelsError
from conftest import RAW_CHANNELS, REDACTED_CHANNELS, SECRET
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

import conda_otel
from conda_otel import capture, core, sources
from conda_otel.core import Dispatcher, Event, Provider
from conda_otel.events import (
    ALL_EVENTS,
    Command,
    EventSelector,
    Outcome,
    matched_events,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any


# Synthetic test data. Every secret contains SECRET.
# channel::package spec; the channel part must not reach package names
SPEC_WITH_CHANNEL = "https://user:SECRET@host.example/t/SECRET/chan::numpy"


# a plugin may raise its own subclass
class FooPluginError(PackagesNotFoundInChannelsError):
    """A plugin's own subclass of the not-found error."""


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace conda's context with the few fields the hooks read."""
    fake = SimpleNamespace(
        plugins=SimpleNamespace(conda_otel=True),
        channels=("conda-forge", "defaults"),
        channel_priority="flexible",
        dry_run=False,
        download_only=False,
    )
    monkeypatch.setattr(capture, "context", fake)
    monkeypatch.setattr(sources, "context", fake)
    return fake


@pytest.fixture
def dispatcher(
    monkeypatch: pytest.MonkeyPatch, in_memory: Provider
) -> Iterator[Dispatcher]:
    """Use a dispatcher with the in-memory provider. Clear the state afterwards."""
    d = Dispatcher([in_memory])
    monkeypatch.setattr(capture, "get_dispatcher", lambda: d)
    yield d
    capture.clear_request()


def sent(provider: Provider) -> list[tuple[Any, dict[str, Any]]]:
    """Return the name and attributes of each event the provider exported."""
    return [
        (r.log_record.body, dict(r.log_record.attributes))
        for r in provider.exporter.get_finished_logs()
    ]


def record(name: str, version: str = "1.0", build: str = "0") -> SimpleNamespace:
    """Make a package record with the fields the hooks read."""
    return SimpleNamespace(name=name, version=version, build=build)


def run_install(requested: tuple[str, ...] | list[str] = ("numpy",)) -> None:
    """Run the capture hooks in the order conda calls them."""
    capture.capture_command("install")
    capture.capture_requested_packages(frozenset(requested), frozenset())
    capture.capture_resolved_packages("repodata.json", (), (record("numpy"),))


def drive_success() -> None:
    """Run a successful install through the hooks."""
    run_install()
    capture.report_success("install")


def drive_packages_not_found() -> None:
    """Run an install that reports missing packages through the hooks."""
    run_install(["nope"])
    exc = PackagesNotFoundInChannelsError(["nope"], ["conda-forge"])
    capture.report_error(
        SimpleNamespace(exc_value=exc, exc_type=type(exc), channels=())
    )


CAPTURE_CASES = {
    "install.success": drive_success,
    "install.packages_not_found": drive_packages_not_found,
}


@pytest.mark.parametrize("name", list(CAPTURE_CASES))
def test_capture_delivers_each_case(
    ctx: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Each capture case reaches a provider subscribed only to its event."""
    command, outcome = name.split(".")
    provider = Provider(
        "in-memory",
        InMemoryLogRecordExporter(),
        events=[EventSelector.exact(Command(command), Outcome(outcome))],
    )
    dispatcher = Dispatcher([provider])
    monkeypatch.setattr(capture, "get_dispatcher", lambda: dispatcher)
    CAPTURE_CASES[name]()
    assert dispatcher.flush() == {"in-memory": True}
    ((event_name, _),) = sent(provider)
    assert event_name == name


def test_capture_cases_match_event_names() -> None:
    """Every supported event has a capture case, with no extra cases."""
    assert set(CAPTURE_CASES) == matched_events([ALL_EVENTS])


def test_success_event(
    ctx: SimpleNamespace, dispatcher: Dispatcher, in_memory: Provider
) -> None:
    """A finished install sends one success event and clears the state."""
    ctx.channels = RAW_CHANNELS
    capture.capture_command("install")
    assert capture.request.channels == REDACTED_CHANNELS
    capture.capture_requested_packages(frozenset({SPEC_WITH_CHANNEL}), frozenset())
    capture.capture_resolved_packages("repodata.json", (), (record("numpy"),))
    capture.report_success("install")
    ((name, attrs),) = sent(in_memory)
    assert name == "install.success"
    assert attrs == {
        "command": "install",
        "channels": tuple(REDACTED_CHANNELS),
        "channel_priority": "flexible",
        "requested_packages": ("numpy",),
        "resolved_packages": ("numpy=1.0=0",),
    }
    assert SECRET not in str(attrs)
    assert not capture.request.active


@pytest.mark.parametrize(
    "channels, expected",
    [
        pytest.param(
            [
                "https://host.example/%74/SECRET/chan",
                "https://host.example/t%2FSECRET/chan",
                "https://host.example/label/a%20b",
            ],
            [
                "https://host.example/chan",
                "https://host.example/chan",
                "https://host.example/label/a%20b",
            ],
            id="encoded-tokens",
        ),
        pytest.param(
            [
                "t/SECRET/user:SECRET@host.example/chan",
                "user:SECRET@t/SECRET/chan",
            ],
            ["host.example/chan", "chan"],
            id="combined-token-userinfo",
        ),
    ],
)
def test_channels_are_redacted_before_transform(
    monkeypatch: pytest.MonkeyPatch,
    dispatcher: Dispatcher,
    in_memory: Provider,
    channels: list[str],
    expected: list[str],
) -> None:
    """Fake credentials are removed from direct events and captured channels."""
    context = Context(
        search_path=(),
        argparse_args=Namespace(channel=channels, override_channels=True),
    )
    monkeypatch.setattr(capture, "context", context)
    monkeypatch.setattr(sources, "context", context)
    seen = []

    def transform(event: Event) -> Event:
        """Record the channels received by the provider."""
        seen.append(event.attributes["channels"])
        return event

    in_memory.transform = transform
    dispatcher.emit(Event("install.success", {"channels": channels}))
    assert seen == [expected]
    run_install()
    assert capture.request.channels == expected
    capture.report_success("install")
    assert seen == [expected, expected]
    assert [attrs["channels"] for _, attrs in sent(in_memory)] == [tuple(expected)] * 2


@pytest.mark.parametrize(
    "raised_class",
    [
        pytest.param(PackagesNotFoundInChannelsError, id="standard"),
        pytest.param(FooPluginError, id="plugin-subclass"),
    ],
)
def test_packages_not_found_event(
    ctx: SimpleNamespace,
    dispatcher: Dispatcher,
    in_memory: Provider,
    raised_class: type[PackagesNotFoundInChannelsError],
) -> None:
    """Missing packages send a known exception name and clear the state."""
    run_install(["nope"])
    exc = raised_class(["nope"], ["conda-forge"])
    event = SimpleNamespace(exc_value=exc, exc_type=type(exc), channels=("from-event",))
    capture.report_error(event)
    ((name, attrs),) = sent(in_memory)
    assert name == "install.packages_not_found"
    assert attrs["channels"] == ("conda-forge", "defaults")
    assert attrs["missing_packages"] == ("nope",)
    assert attrs["requested_packages"] == ("nope",)
    assert attrs["exception_name"] == "PackagesNotFoundInChannelsError"
    assert not capture.request.active


@pytest.mark.parametrize(
    "name", [pytest.param(name, id=name) for name in CAPTURE_CASES]
)
def test_capture_sends_only_declared_attributes(
    ctx: SimpleNamespace,
    dispatcher: Dispatcher,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Capture produces no attribute that the core table does not declare."""
    seen = []
    monkeypatch.setattr(capture, "send", seen.append)
    CAPTURE_CASES[name]()
    assert len(seen) == 1
    for event in seen:
        assert set(event.attributes) <= set(core.ATTRIBUTES)


def test_only_sources_reads_context_channels() -> None:
    """No module except sources.py reads the raw channels."""
    for path in Path(conda_otel.__file__).parent.glob("*.py"):
        if path.name != "sources.py":
            assert "context.channels" not in path.read_text()


def test_other_exception_sends_nothing_and_clears_state(
    ctx: SimpleNamespace, dispatcher: Dispatcher, in_memory: Provider
) -> None:
    """Other errors send nothing, but the state is cleared."""
    run_install()
    exc = RuntimeError("boom")
    capture.report_error(
        SimpleNamespace(exc_value=exc, exc_type=type(exc), channels=())
    )
    assert sent(in_memory) == []
    assert not capture.request.active


def test_setting_off_captures_nothing(
    ctx: SimpleNamespace, dispatcher: Dispatcher, in_memory: Provider
) -> None:
    """With the setting off, nothing is captured or sent."""
    ctx.plugins.conda_otel = False
    run_install()
    capture.report_success("install")
    assert sent(in_memory) == []


def test_no_provider_captures_nothing(
    ctx: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no provider, capture does not start."""
    monkeypatch.setattr(capture, "get_dispatcher", lambda: Dispatcher([]))
    capture.capture_command("install")
    assert not capture.request.active


@pytest.mark.parametrize("flag", ["dry_run", "download_only"])
def test_no_success_event_when_nothing_is_installed(
    ctx: SimpleNamespace, dispatcher: Dispatcher, in_memory: Provider, flag: str
) -> None:
    """A dry run or download-only run sends no success event."""
    run_install()
    setattr(ctx, flag, True)
    capture.report_success("install")
    assert sent(in_memory) == []


@pytest.mark.parametrize(
    "specs, expected",
    [
        (["numpy>=1", "pandas"], ["numpy", "pandas"]),
        (["https://example.com/x-1.0-0.conda"], None),
        (["bad["], None),
        pytest.param(["numpy", "bad["], None, id="mixed-valid-invalid"),
    ],
)
def test_package_names(specs: list[str], expected: list[str] | None) -> None:
    """Names are exact. A URL or a bad name gives None."""
    assert capture.package_names(specs) == expected


def test_invalid_requested_spec_skips_event(
    ctx: SimpleNamespace, dispatcher: Dispatcher, in_memory: Provider
) -> None:
    """One invalid requested spec skips the event, even with a valid spec."""
    run_install(["numpy", "bad["])
    assert capture.request.requested_names is None
    capture.report_success("install")
    assert sent(in_memory) == []
    assert not capture.request.active


def test_failing_provider_does_not_stop_the_command(
    ctx: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error while the providers load must not reach conda."""

    def fail() -> None:
        """Raise like a provider that cannot load."""
        raise RuntimeError("boom")

    monkeypatch.setattr(capture, "get_dispatcher", fail)
    capture.capture_command("install")
    assert not capture.request.active
