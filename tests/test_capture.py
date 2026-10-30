from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from conda.exceptions import PackagesNotFoundInChannelsError
from conda_otel import capture
from conda_otel.core import Dispatcher, Provider


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


def test_success_event(
    ctx: SimpleNamespace, dispatcher: Dispatcher, in_memory: Provider
) -> None:
    """A finished install sends one success event and clears the state."""
    run_install()
    capture.report_success("install")
    ((name, attrs),) = sent(in_memory)
    assert name == "install.success"
    assert attrs == {
        "command": "install",
        "channels": ("conda-forge", "defaults"),
        "channel_priority": "flexible",
        "requested_packages": ("numpy",),
        "resolved_packages": ("numpy=1.0=0",),
    }
    assert not capture.request.active


def test_packages_not_found_event(
    ctx: SimpleNamespace, dispatcher: Dispatcher, in_memory: Provider
) -> None:
    """Missing packages send one event and clear the state."""
    run_install(["nope"])
    exc = PackagesNotFoundInChannelsError(["nope"], ["conda-forge"])
    event = SimpleNamespace(
        exc_value=exc, exc_type=type(exc), channels=("from-event",)
    )
    capture.report_error(event)
    ((name, attrs),) = sent(in_memory)
    assert name == "install.packages_not_found"
    assert attrs["channels"] == ("conda-forge", "defaults")
    assert attrs["missing_packages"] == ("nope",)
    assert attrs["requested_packages"] == ("nope",)
    assert attrs["exception_name"] == "PackagesNotFoundInChannelsError"
    assert not capture.request.active


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
        (["Bad Name!"], None),
    ],
)
def test_package_names(specs: list[str], expected: list[str] | None) -> None:
    """Names are exact. A URL or a bad name gives None."""
    assert capture.package_names(specs) == expected


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
