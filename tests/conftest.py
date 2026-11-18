from __future__ import annotations

import pytest
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

from conda_otel.core import Provider
from conda_otel.events import Command, EventSelector, Outcome


@pytest.fixture
def in_memory() -> Provider:
    """Provider that keeps its records in memory."""
    return Provider(
        "in-memory",
        InMemoryLogRecordExporter(),
        events=[
            EventSelector.exact(Command.INSTALL, Outcome.SUCCESS),
            EventSelector.exact(Command.INSTALL, Outcome.PACKAGES_NOT_FOUND),
        ],
    )
