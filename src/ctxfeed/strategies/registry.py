# strategy registry: maps name -> Strategy instance

from __future__ import annotations

from ctxfeed.strategies.base import Strategy

_REGISTRY: dict[str, Strategy] = {}


def register_strategy(strategy: Strategy) -> None:
    if strategy.name in _REGISTRY:
        raise ValueError(
            f"Strategy '{strategy.name}' is already registered. "
            f"Existing: {_REGISTRY[strategy.name]}, new: {strategy}"
        )
    _REGISTRY[strategy.name] = strategy


def get_strategy(name: str) -> Strategy:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"No strategy registered as '{name}'. Available: {available}"
        )
    return _REGISTRY[name]


def get_all_strategies() -> dict[str, Strategy]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()

