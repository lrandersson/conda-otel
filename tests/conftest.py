from __future__ import annotations

import pytest
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

from conda_otel.core import Provider


@pytest.fixture
def in_memory() -> Provider:
    """Provider that keeps its records in memory."""
    return Provider("in-memory", InMemoryLogRecordExporter())
