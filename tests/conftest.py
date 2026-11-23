from __future__ import annotations

import pytest
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

from conda_otel.core import Provider
from conda_otel.events import Command, EventSelector, Outcome


# Synthetic test data. Every secret contains SECRET.
#: Marker for a substring check that finds any leak.
SECRET = "SECRET"
#: Channels a user may configure. Each item hides one kind of secret.
RAW_CHANNELS = [
    "https://user:SECRET@host.example/chan",  # password in the URL
    "https://host.example:1234/t/SECRET_TO.KEN/chan",  # conda token with "_" and "."
    "https://host.example/chan?token=SECRET",  # token in the query
    "file:///home/SECRET/chan",  # local path
    "conda-forge",  # plain name, kept as is
]
#: RAW_CHANNELS after redaction, item by item.
REDACTED_CHANNELS = [
    "https://host.example/chan",
    "https://host.example:1234/chan",
    "https://host.example/chan",
    "<local>",
    "conda-forge",
]


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
