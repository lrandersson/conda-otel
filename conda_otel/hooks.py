"""Conda plugin registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from conda.common.configuration import PrimitiveParameter
from conda.plugins import hookimpl
from conda.plugins.types import (
    CondaExceptionObserver,
    CondaPostCommand,
    CondaPostSolve,
    CondaPreCommand,
    CondaPreSolve,
    CondaSetting,
)

from conda_otel.capture import (
    capture_command,
    capture_requested_packages,
    capture_resolved_packages,
    report_error,
    report_success,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@hookimpl
def conda_settings() -> Iterator[CondaSetting]:
    """Add the setting that turns conda-otel on or off."""
    yield CondaSetting(
        name="conda_otel",
        description="Whether conda-otel captures events and sends them to providers",
        parameter=PrimitiveParameter(True, element_type=bool),
    )


@hookimpl
def conda_pre_commands() -> Iterator[CondaPreCommand]:
    """Start the capture for `conda install`."""
    yield CondaPreCommand(
        name="conda-otel-pre-command",
        action=capture_command,
        run_for={"install"},
    )


@hookimpl
def conda_pre_solves() -> Iterator[CondaPreSolve]:
    """Capture the requested packages."""
    yield CondaPreSolve(name="conda-otel-pre-solve", action=capture_requested_packages)


@hookimpl
def conda_post_solves() -> Iterator[CondaPostSolve]:
    """Capture the resolved packages."""
    yield CondaPostSolve(name="conda-otel-post-solve", action=capture_resolved_packages)


@hookimpl
def conda_post_commands() -> Iterator[CondaPostCommand]:
    """Send the success event."""
    yield CondaPostCommand(
        name="conda-otel-post-command",
        action=report_success,
        run_for={"install"},
    )


@hookimpl
def conda_exception_observers() -> Iterator[CondaExceptionObserver]:
    """Send the packages-not-found event and clear the state on any error."""
    yield CondaExceptionObserver(
        name="conda-otel",
        hook=report_error,
        # Observe BaseException so report_error() can always clear the state.
        watch_for={"BaseException"},
    )
