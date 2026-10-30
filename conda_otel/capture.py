"""Capture conda install data and send it as neutral events."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from conda.base.context import context
from conda.exceptions import InvalidMatchSpec, PackagesNotFoundInChannelsError
from conda.models.match_spec import MatchSpec

from conda_otel.core import Dispatcher, Event, load_providers

if TYPE_CHECKING:
    from conda.models.records import PackageRecord
    from conda.plugins.types import CondaExceptionEvent

logger = logging.getLogger(__name__)


class EventName(StrEnum):
    """Neutral event names. Providers map them to their own names."""

    PACKAGES_NOT_FOUND = "install.packages_not_found"
    SUCCESS = "install.success"


@dataclass
class InstallRequest:
    """State of the current `conda install` command.

    Filled by the pre-command, pre-solve and post-solve hooks.
    Read and cleared by the exception observer or the post-command hook.
    """

    #: True when the setting is on and a provider started.
    #: When False, the later hooks do nothing.
    active: bool = False
    channels: list[str] | None = None
    requested_names: list[str] | None = None
    resolved_packages: list[str] | None = None


request = InstallRequest()


@lru_cache(maxsize=1)
def get_dispatcher() -> Dispatcher:
    """Load the providers once per process."""
    return Dispatcher(load_providers(context))


def package_names(specs: list[Any]) -> list[str] | None:
    """Return the sorted, unique exact names of the specs.

    Return None if a spec is a URL, is invalid, or has a name outside `[a-z0-9_.-]`.
    """
    names = set()
    for value in specs:
        try:
            spec = MatchSpec(value)
        except InvalidMatchSpec:
            return None
        name = spec.get_exact_value("name")
        if (
            spec.get_raw_value("url")
            or not name
            or not re.fullmatch(r"[a-z0-9_.-]+", name)
        ):
            return None
        names.add(name)
    return sorted(names)


def clear_request() -> None:
    """Reset the captured state."""
    request.active = False
    request.channels = None
    request.requested_names = None
    request.resolved_packages = None


def capture_command(command: str) -> None:
    """Pre-command hook: start a new capture.

    Capture only if the setting is on and a provider started.
    """
    clear_request()
    # Note: this hook must never stop the command, for example if a provider fails.
    try:
        if context.plugins.conda_otel and get_dispatcher():
            request.active = True
            # Both events use the configured channels.
            request.channels = list(context.channels)
    except Exception:
        logger.debug("Telemetry capture is off: it failed to start.", exc_info=True)
        clear_request()


def capture_requested_packages(specs_to_add, _specs_to_remove) -> None:
    """Pre-solve hook: save the requested package names."""
    # Note: this hook must never stop the command.
    try:
        if request.active:
            request.requested_names = package_names(list(specs_to_add))
    except Exception:
        logger.debug("Failed to capture the requested packages.", exc_info=True)
        clear_request()


def capture_resolved_packages(
    _repodata_fn: str,
    _unlink_precs: tuple[PackageRecord, ...],
    link_precs: tuple[PackageRecord, ...],
) -> None:
    """Post-solve hook: save the packages that will be installed."""
    # Note: this hook must never stop the command.
    try:
        if request.active:
            request.resolved_packages = [
                f"{record.name}={record.version}={record.build}"
                for record in link_precs
            ]
    except Exception:
        logger.debug("Failed to capture the resolved packages.", exc_info=True)
        clear_request()


def send(event: Event) -> None:
    """Send the event to all providers and flush."""
    dispatcher = get_dispatcher()
    dispatcher.emit(event)
    undelivered = [name for name, ok in dispatcher.flush().items() if not ok]
    if undelivered:
        logger.debug("Event %s was not delivered to: %s", event.name, undelivered)


def report_error(event: CondaExceptionEvent) -> None:
    """Exception observer: send an event when packages are not found.

    Clear the state on every error.
    """
    try:
        if not request.active or request.requested_names is None:
            return
        if not isinstance(event.exc_value, PackagesNotFoundInChannelsError):
            return
        missing = package_names(list(event.exc_value.packages))
        if missing is None:
            return
        send(
            Event(
                EventName.PACKAGES_NOT_FOUND,
                {
                    "command": "install",
                    "channels": request.channels,
                    "channel_priority": str(context.channel_priority),
                    "requested_packages": request.requested_names,
                    "missing_packages": missing,
                    "exception_name": event.exc_type.__name__,
                },
            )
        )
    except Exception as exc:
        logger.debug("Failed to send event for %s", event.exc_type, exc_info=exc)
    finally:
        # Post-command hooks do not run on failure, so clear the state here.
        clear_request()


def report_success(command: str) -> None:
    """Post-command hook: send an event for a finished install."""
    try:
        # These flags reach this hook when nothing was installed.
        if context.dry_run or context.download_only:
            return
        if (
            not request.active
            or request.requested_names is None
            or request.resolved_packages is None
            or request.channels is None
        ):
            return
        send(
            Event(
                EventName.SUCCESS,
                {
                    "command": "install",
                    "channels": request.channels,
                    "channel_priority": str(context.channel_priority),
                    "requested_packages": request.requested_names,
                    "resolved_packages": request.resolved_packages,
                },
            )
        )
    except Exception as exc:
        logger.debug("Failed to send event for %s", command, exc_info=exc)
    finally:
        clear_request()
