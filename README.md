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
        from conda_otel.events import Command, EventSelector, Outcome

        yield Provider(
            name="my-provider",
            exporter=make_exporter(context),
            events=[EventSelector.exact(Command.INSTALL, Outcome.SUCCESS)],
        )
```

`Provider` takes these fields:

- `name` identifies the provider in flush results and debug logs.
- `exporter` is the OpenTelemetry `LogRecordExporter` that sends the log records.
- `events` is required and keyword-only. It lists the `EventSelector` values for the events the provider receives. See [Subscriptions](#subscriptions).
- `resource_attributes` (optional) are OpenTelemetry resource attributes for this provider's log records.
- `transform` (optional) takes an `Event` and returns a changed `Event`, or `None` to drop the event for this provider.

Users enable the provider in `.condarc`:

```yaml
plugins:
  my_provider: true
```

or with the environment variable `CONDA_PLUGINS_MY_PROVIDER=true`.
The global `conda_otel` setting is on by default. Set it to `false` to turn off all telemetry.

## Subscriptions

`events` is a list of selectors from `conda_otel.events`. A provider receives only the events that its selectors match.

An exact selector matches one registered pair:

```python
EventSelector.exact(Command.INSTALL, Outcome.SUCCESS)
```

`EventSelector.for_command(...)` matches every outcome of that command, including outcomes added later, and `EventSelector.for_outcome(...)` matches that outcome for every command. `[ALL_EVENTS]` matches every event in the registry, including events added later. Prefer exact selectors, otherwise a new event may reach your provider with future updates.

Use a list and not a bare selector (`events=EventSelector.exact(...)` is rejected).

To check what a declaration matches, use `matched_events()`. It returns the event names as a `frozenset`:

```python
>>> from conda_otel.events import Command, EventSelector, Outcome, matched_events
>>> matched_events([EventSelector.exact(Command.INSTALL, Outcome.SUCCESS)])
frozenset({'install.success'})
```

If the declaration is invalid, conda-otel logs a warning and skips that provider; the others keep running.

## What providers receive

The core checks these attributes before calling any provider transform.

| Attribute | Events | What conda-otel does |
| --- | --- | --- |
| `command` | Both install events | Allows only `install`. |
| `channels` | Both install events | Removes URL userinfo, `/t/<token>` pairs, queries and fragments; replaces local paths with `<local>` and unparsable URLs with `<invalid>`. Keeps channel names and other URL parts. |
| `channel_priority` | Both install events | Allows only `strict`, `flexible` and `disabled`. |
| `requested_packages` | Both install events | Reduces specs to exact names matching `[a-z0-9_.-]+`; drops the list if any spec fails. Checks names again before dispatch. |
| `resolved_packages` | `install.success` | Keeps `name=version=build` values from package metadata unchanged. |
| `missing_packages` | `install.packages_not_found` | Applies the same exact-name checks as `requested_packages`. |
| `exception_name` | `install.packages_not_found` | Allows only `PackagesNotFoundInChannelsError`; maps an unknown subclass to its nearest known parent. |

Undeclared attributes are dropped.
Providers also get the full conda `context` in `conda_otel_providers` and must not send raw values from it.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
