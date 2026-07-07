"""Adapter registration and lookup (spec §7.1). Adapters self-register under
(port, name) at import time; the factory resolves names from config here.
Adding a new algorithm = one new decorated class, nothing else changes."""

_REGISTRY: dict[type, dict[str, type]] = {}


class RegistryError(Exception):
    pass


def register(port: type, name: str):
    """Class decorator: `@register(Scorer, "mderank")`."""

    def decorator(adapter_cls: type) -> type:
        if not issubclass(adapter_cls, port):
            raise RegistryError(
                f"{adapter_cls.__name__} does not implement {port.__name__}"
            )
        adapters = _REGISTRY.setdefault(port, {})
        if name in adapters:
            raise RegistryError(
                f"duplicate adapter {name!r} for port {port.__name__}"
            )
        adapters[name] = adapter_cls
        return adapter_cls

    return decorator


def lookup(port: type, name: str) -> type:
    """Resolve a registered adapter class by (port, name), as used by the
    factory when wiring config-named adapters."""
    adapters = _REGISTRY.get(port, {})
    if name not in adapters:
        known = ", ".join(sorted(adapters)) or "none registered"
        raise RegistryError(
            f"no adapter {name!r} for port {port.__name__} (known: {known})"
        )
    return adapters[name]


def available(port: type) -> dict[str, type]:
    """Return a copy of the name -> adapter class mapping registered for a
    port, e.g. for listing choices in config validation errors."""
    return dict(_REGISTRY.get(port, {}))
