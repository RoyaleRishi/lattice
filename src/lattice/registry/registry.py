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
    adapters = _REGISTRY.get(port, {})
    if name not in adapters:
        known = ", ".join(sorted(adapters)) or "none registered"
        raise RegistryError(
            f"no adapter {name!r} for port {port.__name__} (known: {known})"
        )
    return adapters[name]


def available(port: type) -> dict[str, type]:
    return dict(_REGISTRY.get(port, {}))
