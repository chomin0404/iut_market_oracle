"""Pre-built BayesianNetwork registry for named, reusable networks.

Named networks avoid sending the full network definition on every request.
All registered networks are read-only shared instances; :func:`infer` is safe
to call concurrently because :meth:`BayesianNetwork.update` save/restores
evidence internally.
"""

from __future__ import annotations

from bayesian.network import BayesianNetwork
from bayesian.water_demand_net import build_fukuoka_water_demand_net

_REGISTRY: dict[str, tuple[str, BayesianNetwork]] = {}


def _register(name: str, description: str, net: BayesianNetwork) -> None:
    _REGISTRY[name] = (description, net)


def get_network(name: str) -> BayesianNetwork | None:
    """Return the named network, or None if not found."""
    entry = _REGISTRY.get(name)
    return entry[1] if entry else None


def list_networks() -> list[dict[str, str]]:
    """Return all registered networks sorted by name."""
    return [{"name": k, "description": v[0]} for k, v in sorted(_REGISTRY.items())]


# ---------------------------------------------------------------------------
# Built-in networks
# ---------------------------------------------------------------------------

_register(
    "fukuoka_water_demand",
    "福岡市水道需要予測ネット — season, day_type → demand_level",
    build_fukuoka_water_demand_net(),
)
