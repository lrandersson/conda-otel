from __future__ import annotations

from enum import StrEnum
from itertools import product

import pytest

from conda_otel import events
from conda_otel.events import (
    ALL_EVENTS,
    Command,
    EventName,
    EventSelector,
    Outcome,
    matched_events,
)


# FutureCommand and FutureOutcome (below) are used to simulate a future release of
# conda-otel to verify the behavior is as expected for providers who opt-in
# only for a subset of certain commands/outcome.
class FutureCommand(StrEnum):
    INSTALL = "install"
    EXAMPLE = "example"  # Simulate a new command without using a real conda command.


class FutureOutcome(StrEnum):
    SUCCESS = "success"
    PACKAGES_NOT_FOUND = "packages_not_found"
    FAILED = "failed"


@pytest.fixture
def future_event_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in test-only enums and keep the production pairs initially."""
    monkeypatch.setattr(events, "Command", FutureCommand)
    monkeypatch.setattr(events, "Outcome", FutureOutcome)
    monkeypatch.setattr(
        events,
        "SUPPORTED_COMMAND_OUTCOME_PAIRS",
        frozenset(
            {
                (FutureCommand.INSTALL, FutureOutcome.SUCCESS),
                (FutureCommand.INSTALL, FutureOutcome.PACKAGES_NOT_FOUND),
            }
        ),
    )


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        pytest.param(
            EventSelector.exact(Command.INSTALL, Outcome.SUCCESS),
            {"install.success"},
            id="exact",
        ),
        pytest.param(
            EventSelector.for_command(Command.INSTALL),
            {"install.success", "install.packages_not_found"},
            id="by-command",
        ),
        pytest.param(
            ALL_EVENTS,
            {"install.success", "install.packages_not_found"},
            id="all-events",
        ),
    ],
)
def test_selector_matches_supported_events(
    selector: EventSelector, expected: set[str]
) -> None:
    """Check which event names a single selector returns."""
    assert matched_events([selector]) == expected


def test_for_outcome_matches_all_commands_with_outcome(
    future_event_catalog: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outcome selector matches that outcome for every command."""
    monkeypatch.setattr(
        events,
        "SUPPORTED_COMMAND_OUTCOME_PAIRS",
        frozenset(
            {
                (FutureCommand.INSTALL, FutureOutcome.SUCCESS),
                (FutureCommand.EXAMPLE, FutureOutcome.SUCCESS),
                (FutureCommand.EXAMPLE, FutureOutcome.FAILED),
            }
        ),
    )
    assert matched_events([EventSelector.for_outcome(FutureOutcome.SUCCESS)]) == {
        "install.success",
        "example.success",
    }


@pytest.mark.parametrize(
    ("selectors", "expected"),
    [
        pytest.param(
            [ALL_EVENTS, EventSelector.exact(Command.INSTALL, Outcome.SUCCESS)],
            {"install.success", "install.packages_not_found"},
            id="all-plus-exact",
        ),
        pytest.param([], set(), id="empty"),
        # All three selectors match install.success.
        pytest.param(
            [
                EventSelector.exact(Command.INSTALL, Outcome.SUCCESS),
                EventSelector.for_command(Command.INSTALL),
                EventSelector.for_outcome(Outcome.SUCCESS),
            ],
            {"install.success", "install.packages_not_found"},
            id="overlapping",
        ),
    ],
)
def test_matched_events_combines_selectors(
    selectors: list[EventSelector], expected: set[str]
) -> None:
    """Return each matching event once, even if several selectors match it."""
    assert matched_events(selectors) == expected


def test_event_name_literal_values() -> None:
    """Check the exact names used for install events."""
    assert EventName.SUCCESS == "install.success"
    assert EventName.PACKAGES_NOT_FOUND == "install.packages_not_found"


def test_event_names_match_supported_pairs() -> None:
    """Every supported pair has an EventName, and no EventName is stale."""
    names = {e.value for e in EventName}
    supported_names = matched_events([ALL_EVENTS])
    assert names == supported_names


def test_supported_pairs_are_strict_subset_of_cross_product(
    future_event_catalog: None,
) -> None:
    """Not every command and outcome pair is supported."""
    cross_product = frozenset(product(FutureCommand, FutureOutcome))
    assert events.SUPPORTED_COMMAND_OUTCOME_PAIRS < cross_product


def test_unsupported_exact_pair_raises(future_event_catalog: None) -> None:
    """An exact selector for an unsupported pair is rejected."""
    with pytest.raises(ValueError):
        matched_events(
            [
                EventSelector.for_command(FutureCommand.INSTALL),
                EventSelector.exact(FutureCommand.EXAMPLE, FutureOutcome.SUCCESS),
            ]
        )


def test_for_outcome_with_no_matches(future_event_catalog: None) -> None:
    """A wildcard with no matching pairs is valid and matches nothing."""
    assert (
        matched_events([EventSelector.for_outcome(FutureOutcome.FAILED)]) == frozenset()
    )


def test_growth_only_adds_intended_matches(
    future_event_catalog: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Newly supported pairs match only the applicable wildcards."""
    exact = [EventSelector.exact(FutureCommand.INSTALL, FutureOutcome.SUCCESS)]
    by_command = [EventSelector.for_command(FutureCommand.INSTALL)]
    by_outcome = [EventSelector.for_outcome(FutureOutcome.SUCCESS)]
    everything = [ALL_EVENTS]

    before = (
        matched_events(exact),
        matched_events(by_command),
        matched_events(by_outcome),
        matched_events(everything),
    )

    monkeypatch.setattr(
        events,
        "SUPPORTED_COMMAND_OUTCOME_PAIRS",
        frozenset(
            {
                (FutureCommand.INSTALL, FutureOutcome.SUCCESS),
                (FutureCommand.INSTALL, FutureOutcome.PACKAGES_NOT_FOUND),
                (FutureCommand.INSTALL, FutureOutcome.FAILED),
                (FutureCommand.EXAMPLE, FutureOutcome.SUCCESS),
                (FutureCommand.EXAMPLE, FutureOutcome.FAILED),
            }
        ),
    )

    after = (
        matched_events(exact),
        matched_events(by_command),
        matched_events(by_outcome),
        matched_events(everything),
    )

    assert after[0] == before[0]
    assert after[1] == before[1] | {"install.failed"}
    assert after[2] == before[2] | {"example.success"}
    assert after[3] == before[3] | {
        "install.failed",
        "example.success",
        "example.failed",
    }


@pytest.mark.parametrize(
    "declaration",
    [
        pytest.param("", id="empty-string"),
        pytest.param(b"", id="empty-bytes"),
        pytest.param({}, id="mapping"),
        pytest.param(42, id="not-iterable"),
        pytest.param(ALL_EVENTS, id="bare-selector"),
        pytest.param([ALL_EVENTS, "install.success"], id="str-entry"),
        pytest.param(
            [EventSelector.exact(Command.INSTALL, "success")], id="plain-str-outcome"
        ),
        pytest.param([EventSelector.for_command(None)], id="command-without-command"),
        pytest.param(
            [EventSelector.exact(Outcome.SUCCESS, Command.INSTALL)],
            id="swapped-arguments",
        ),
        pytest.param(
            [ALL_EVENTS, EventSelector.exact(Command.INSTALL, "success")],
            id="valid-then-malformed",
        ),
        pytest.param([EventSelector("bogus")], id="unknown-kind"),
        pytest.param(
            [EventSelector("command", Command.INSTALL, Outcome.SUCCESS)],
            id="command-with-outcome",
        ),
        pytest.param(
            [EventSelector("outcome", Command.INSTALL, Outcome.SUCCESS)],
            id="outcome-with-command",
        ),
        pytest.param(
            [EventSelector("all", Command.INSTALL)],
            id="all-with-command",
        ),
    ],
)
def test_invalid_declaration_raises(declaration: object) -> None:
    """Malformed declarations raise ValueError."""
    with pytest.raises(ValueError):
        matched_events(declaration)
