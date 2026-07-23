"""Strategy registry: name → Strategy instance lookup.

Provides a central place to register and retrieve strategy instances
by name. The evaluation runner uses this to iterate over all strategies.

Usage:
    from ctxfeed.strategies.registry import get_strategy, get_all_strategies

    s1 = get_strategy("s1_naive")
    all_strategies = get_all_strategies()
"""

from __future__ import annotations

from ctxfeed.strategies.base import Strategy


# Global registry populated by register_strategy()
_REGISTRY: dict[str, Strategy] = {}


def register_strategy(strategy: Strategy) -> None:
    """Register a strategy instance by its name.

    Raises:
        ValueError: If a strategy with the same name is already registered.
    """
    if strategy.name in _REGISTRY:
        raise ValueError(
            f"Strategy '{strategy.name}' is already registered. "
            f"Existing: {_REGISTRY[strategy.name]}, new: {strategy}"
        )
    _REGISTRY[strategy.name] = strategy


def get_strategy(name: str) -> Strategy:
    """Retrieve a registered strategy by name.

    Raises:
        KeyError: If no strategy is registered under *name*.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"No strategy registered as '{name}'. Available: {available}"
        )
    return _REGISTRY[name]


def get_all_strategies() -> dict[str, Strategy]:
    """Return a copy of the full strategy registry."""
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Remove all registered strategies. Used in tests."""
    _REGISTRY.clear()
