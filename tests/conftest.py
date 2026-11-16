from __future__ import annotations

import pytest
from conda_otel.core import Provider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter


@pytest.fixture
def in_memory() -> Provider:
    """Provider that keeps its records in memory."""
    return Provider("in-memory", InMemoryLogRecordExporter())
