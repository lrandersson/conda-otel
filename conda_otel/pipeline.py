"""Private OTel pipeline for one provider. Imports the OTel SDK."""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry._logs import LogRecord
from opentelemetry.sdk._logs import LoggerProvider, LogRecordLimits, ReadWriteLogRecord
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
)
from opentelemetry.sdk.resources import Resource

from conda_otel.core import Event, Provider, ignore_otel_environment

# A user must not see telemetry errors, for example on a blocked network.
# Conda prints root logger records to stderr, so stop OTel records here.
# The NullHandler stops Python from printing records that have no handler.
otel_logger = logging.getLogger("opentelemetry")
otel_logger.addHandler(logging.NullHandler())
otel_logger.propagate = False

#: Explicit record limits. They stop `OTEL_ATTRIBUTE_*` variables from
#: changing the output. Providers cap their own values.
LIMITS = LogRecordLimits(max_attributes=1_000, max_attribute_length=1_000_000)


class TrackingExporter(LogRecordExporter):
    """Wrap an exporter and remember if the last export worked.

    The batch processor hides export errors. This wrapper makes them visible.
    """

    def __init__(self, exporter: LogRecordExporter):
        """Wrap the exporter."""
        self.exporter = exporter
        #: None until the first export. Then the result of the last export.
        self.delivered: bool | None = None

    def export(self, batch: Any) -> LogRecordExportResult:
        """Export the batch and save the result."""
        try:
            result = self.exporter.export(batch)
        except Exception:
            self.delivered = False
            raise
        self.delivered = result == LogRecordExportResult.SUCCESS
        return result

    def force_flush(self, timeout_millis: int = 10_000) -> bool:
        """Flush the wrapped exporter."""
        return self.exporter.force_flush(timeout_millis)

    def shutdown(self) -> None:
        """Shut down the wrapped exporter."""
        self.exporter.shutdown()


class Pipeline:
    """One private logger with its own batch processor for one provider."""

    def __init__(self, provider: Provider):
        """Build the private logger and batch processor for the provider."""
        self.provider = provider
        self.exporter = TrackingExporter(provider.exporter)
        with ignore_otel_environment():
            self.logger_provider = LoggerProvider(
                resource=Resource.create(provider.resource_attributes),
                shutdown_on_exit=False,
            )
            self.logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(self.exporter)
            )
        self.logger = self.logger_provider.get_logger("conda_otel")

    def emit(self, event: Event) -> None:
        """Queue one event for export."""
        record = ReadWriteLogRecord(
            log_record=LogRecord(
                body=event.name,
                event_name=event.name,
                attributes=event.attributes,
            ),
            resource=self.logger.resource,
            limits=LIMITS,
        )
        self.logger.emit(record)
