# conda-otel

A [conda](https://docs.conda.io) plugin that collects data about `conda install` and sends it to provider packages. It is vendor-neutral.

> **Status:** proof of concept. Do not use in production.

## Register a provider

A provider is a conda plugin that implements the `conda_otel_providers` hook.
Register the hooks module in `pyproject.toml`:

```toml
[project.entry-points.conda]
my-provider = "my_provider.hooks"
```

Example `my_provider/hooks.py`:

```python
from conda.common.configuration import PrimitiveParameter
from conda.plugins import hookimpl
from conda.plugins.types import CondaSetting


@hookimpl
def conda_settings():
    yield CondaSetting(
        name="my_provider",
        description="Whether to send conda-otel events to my provider",
        parameter=PrimitiveParameter(False, element_type=bool),
    )


@hookimpl(optionalhook=True)
def conda_otel_providers(context):
    if context.plugins.my_provider:
        from my_provider.exporter import make_exporter

        from conda_otel.core import Provider

        yield Provider(name="my-provider", exporter=make_exporter(context))
```

`Provider` takes these fields:

- `name` identifies the provider in flush results and debug logs.
- `exporter` is the OpenTelemetry `LogRecordExporter` that sends the log records.
- `resource_attributes` (optional) are OpenTelemetry resource attributes for this provider's log records.
- `transform` (optional) takes an `Event` and returns a changed `Event`, or `None` to drop the event for this provider.

Users enable the provider in `.condarc`:

```yaml
plugins:
  my_provider: true
```

or with the environment variable `CONDA_PLUGINS_MY_PROVIDER=true`.
The global `conda_otel` setting is on by default. Set it to `false` to turn off all telemetry.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
