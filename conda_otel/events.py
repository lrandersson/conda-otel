"""Event names and subscription selectors for providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal


class Command(StrEnum):
    """Supported conda commands that providers can subscribe to."""

    INSTALL = "install"


class Outcome(StrEnum):
    """The outcome of a command."""

    SUCCESS = "success"
    PACKAGES_NOT_FOUND = "packages_not_found"


class EventName(StrEnum):
    """Neutral event names. Values come from the Command and Outcome enums."""

    PACKAGES_NOT_FOUND = f"{Command.INSTALL}.{Outcome.PACKAGES_NOT_FOUND}"
    SUCCESS = f"{Command.INSTALL}.{Outcome.SUCCESS}"


#: Supported (command, outcome) pairs.
SUPPORTED_COMMAND_OUTCOME_PAIRS: frozenset[tuple[Command, Outcome]] = frozenset(
    {
        (Command.INSTALL, Outcome.SUCCESS),
        (Command.INSTALL, Outcome.PACKAGES_NOT_FOUND),
    }
)


@dataclass(frozen=True)
class EventSelector:
    """Select supported telemetry events by command, outcome, or both.

    Use `exact`, `for_command`, or `for_outcome` to build a selector,
    or use `ALL_EVENTS` to select every supported event.
    """

    kind: Literal["exact", "command", "outcome", "all"]
    command: Command | None = None
    outcome: Outcome | None = None

    @classmethod
    def exact(cls, command: Command, outcome: Outcome) -> EventSelector:
        """Match one registered (command, outcome) pair."""
        return cls("exact", command, outcome)

    @classmethod
    def for_command(cls, command: Command) -> EventSelector:
        """Match every outcome of one command, including future ones."""
        return cls("command", command)

    @classmethod
    def for_outcome(cls, outcome: Outcome) -> EventSelector:
        """Match one outcome of every command, including future ones."""
        return cls("outcome", outcome=outcome)


ALL_EVENTS = EventSelector("all")


def _check(selector: EventSelector) -> None:
    """Raise ValueError when the selector fields do not fit its kind."""
    kind, command, outcome = selector.kind, selector.command, selector.outcome
    if kind == "exact":
        valid = isinstance(command, Command) and isinstance(outcome, Outcome)
    elif kind == "command":
        valid = isinstance(command, Command) and outcome is None
    elif kind == "outcome":
        valid = command is None and isinstance(outcome, Outcome)
    elif kind == "all":
        valid = command is None and outcome is None
    else:
        valid = False
    if not valid:
        raise ValueError(f"invalid EventSelector: {selector!r}")
    if kind == "exact" and (command, outcome) not in SUPPORTED_COMMAND_OUTCOME_PAIRS:
        raise ValueError(f"unsupported event: {command}.{outcome}")


def _matches(selector: EventSelector, command: Command, outcome: Outcome) -> bool:
    """Return True when the selector matches the pair."""
    if selector.kind == "all":
        return True
    if selector.kind == "exact":
        return selector.command == command and selector.outcome == outcome
    if selector.kind == "command":
        return selector.command == command
    return selector.outcome == outcome


def matched_events(events: Iterable[EventSelector]) -> frozenset[str]:
    """Return the names of supported events that any selector matches.

    Raises ValueError when the declaration is malformed.
    """
    if isinstance(events, EventSelector):
        raise ValueError("use events=[selector] instead of events=selector")
    if isinstance(events, str | bytes | Mapping):
        raise ValueError(f"events must be a list of EventSelector, not {type(events)}")
    try:
        selectors = list(events)
    except TypeError:
        raise ValueError(f"events must be iterable, not {type(events)}") from None

    for selector in selectors:
        if not isinstance(selector, EventSelector):
            raise ValueError(f"expected EventSelector, got {selector!r}")
        _check(selector)

    return frozenset(
        f"{command.value}.{outcome.value}"
        for command, outcome in SUPPORTED_COMMAND_OUTCOME_PAIRS
        if any(_matches(s, command, outcome) for s in selectors)
    )
