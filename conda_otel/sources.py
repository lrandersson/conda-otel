"""The only module that reads sensitive values from the conda context."""

from conda.base.context import context

from conda_otel.core import redact_channel


def configured_channels() -> list[str]:
    """Return configured channels with credentials and local paths removed."""
    return [redact_channel(c) for c in context.channels]
