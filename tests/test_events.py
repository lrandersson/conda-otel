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


class FutureCommand(StrEnum):
    INSTALL = "install"
    CREATE = "create"


class FutureOutcome(StrEnum):
    SUCCESS = "success"
    PACKAGES_NOT_FOUND = "packages_not_found"
    FAILED = "failed"


@pytest.fixture
def future_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in test-only enums. The registry starts with the production pairs."""
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


def test_exact_selector_matches_one_event() -> None:
    """An exact selector matches only its own pair."""
    assert matched_events([EventSelector.exact(Command.INSTALL, Outcome.SUCCESS)]) == {
        "install.success"
    }


def test_for_command_matches_all_outcomes_of_command() -> None:
    """A command selector matches every outcome of that command."""
    assert matched_events([EventSelector.for_command(Command.INSTALL)]) == {
        "install.success",
        "install.packages_not_found",
    }


def test_for_outcome_matches_all_commands_with_outcome(
    future_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outcome selector matches that outcome for every command."""
    monkeypatch.setattr(
        events,
        "SUPPORTED_COMMAND_OUTCOME_PAIRS",
        frozenset(
            {
                (FutureCommand.INSTALL, FutureOutcome.SUCCESS),
                (FutureCommand.CREATE, FutureOutcome.SUCCESS),
                (FutureCommand.CREATE, FutureOutcome.FAILED),
            }
        ),
    )
    assert matched_events([EventSelector.for_outcome(FutureOutcome.SUCCESS)]) == {
        "install.success",
        "create.success",
    }


def test_all_events_matches_every_registry_pair() -> None:
    """The wildcard matches every registry pair."""
    assert matched_events([ALL_EVENTS]) == {
        "install.success",
        "install.packages_not_found",
    }


def test_overlapping_selectors_are_deduplicated() -> None:
    """Pairs matched by several selectors appear once in the result."""
    selectors = [
        EventSelector.exact(Command.INSTALL, Outcome.SUCCESS),
        EventSelector.for_command(Command.INSTALL),
        EventSelector.for_outcome(Outcome.SUCCESS),
    ]
    assert matched_events(selectors) == {
        "install.success",
        "install.packages_not_found",
    }


def test_all_events_mixed_with_exact_selector() -> None:
    """The wildcard combines with exact selectors without changing the result."""
    selectors = [ALL_EVENTS, EventSelector.exact(Command.INSTALL, Outcome.SUCCESS)]
    assert matched_events(selectors) == {
        "install.success",
        "install.packages_not_found",
    }


def test_empty_declaration_matches_nothing() -> None:
    """An empty list matches no events."""
    assert matched_events([]) == frozenset()


def test_event_name_literal_values() -> None:
    """EventName values are the wire-facing neutral names."""
    assert EventName.SUCCESS == "install.success"
    assert EventName.PACKAGES_NOT_FOUND == "install.packages_not_found"


def test_event_names_match_registry() -> None:
    """Every registry pair has an EventName, and no EventName is stale."""
    names = {e.value for e in EventName}
    registry = matched_events([ALL_EVENTS])
    missing = sorted(registry - names)
    extra = sorted(names - registry)
    assert names == registry, f"missing from EventName: {missing}; extra: {extra}"


def test_supported_pairs_are_strict_subset_of_cross_product(
    future_registry: None,
) -> None:
    """Not every command and outcome pair is supported."""
    cross_product = frozenset(product(FutureCommand, FutureOutcome))
    assert events.SUPPORTED_COMMAND_OUTCOME_PAIRS < cross_product


def test_unregistered_exact_pair_raises(future_registry: None) -> None:
    """An exact selector for an unsupported pair is rejected."""
    with pytest.raises(ValueError):
        matched_events(
            [
                EventSelector.for_command(FutureCommand.INSTALL),
                EventSelector.exact(FutureCommand.CREATE, FutureOutcome.SUCCESS),
            ]
        )


def test_for_outcome_without_registered_pairs_is_empty(future_registry: None) -> None:
    """A wildcard with no matching pairs is valid and matches nothing."""
    assert (
        matched_events([EventSelector.for_outcome(FutureOutcome.FAILED)]) == frozenset()
    )


def test_growth_only_adds_intended_matches(
    future_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New registry pairs reach only the wildcards that should match them."""
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
                (FutureCommand.CREATE, FutureOutcome.SUCCESS),
                (FutureCommand.CREATE, FutureOutcome.FAILED),
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
    assert after[2] == before[2] | {"create.success"}
    assert after[3] == before[3] | {"install.failed", "create.success", "create.failed"}


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
