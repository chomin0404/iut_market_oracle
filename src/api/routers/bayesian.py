"""Bayesian update and network inference endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException

from api.dependencies import make_api_key_dep
from api.schemas.bayesian import (
    CptDef,
    InferRequest,
    InferResponse,
    NamedInferRequest,
    NetworkInfo,
    UpdateRequest,
)
from bayesian.network import BayesianNetwork, infer
from bayesian.registry import get_network, list_networks
from bayesian.updater import update
from schemas import PosteriorSummary

router = APIRouter()
_logger = logging.getLogger(__name__)

# Inference endpoints are compute-intensive; require X-API-Key when ORACLE_API_KEY is set.
_require_oracle_key = make_api_key_dep("X-API-Key", "ORACLE_API_KEY")

_BAYESIAN_EXAMPLES = {
    "beta_prior": {
        "summary": "Beta prior (conversion rate)",
        "value": {
            "prior": {"distribution": "beta", "params": {"alpha": 2.0, "beta": 18.0}},
            "evidence": [
                {"source": "obs_1", "kind": "observation", "value": 0.15, "weight": 1.0},
                {"source": "obs_2", "kind": "observation", "value": 0.20, "weight": 1.0},
            ],
        },
    },
    "normal_prior": {
        "summary": "Normal prior (return estimate)",
        "value": {
            "prior": {"distribution": "normal", "params": {"mu": 0.05, "sigma": 0.02}},
            "evidence": [
                {"source": "q1_return", "kind": "observation", "value": 0.07, "weight": 1.0},
            ],
        },
    },
}

_INFER_EXAMPLES = {
    "economy_regime": {
        "summary": "economy → regime (expansion observed)",
        "value": {
            "nodes": [
                {"id": "economy", "states": ["expansion", "recession"]},
                {"id": "regime", "states": ["bull", "bear", "neutral"]},
            ],
            "edges": [{"parent": "economy", "child": "regime"}],
            "priors": [{"node": "economy", "probs": [0.70, 0.30]}],
            "cpts": [
                {
                    "node": "regime",
                    "rows": [
                        {"parents": ["expansion"], "probs": [0.60, 0.10, 0.30]},
                        {"parents": ["recession"], "probs": [0.20, 0.60, 0.20]},
                    ],
                }
            ],
            "query": "regime",
            "evidence": {"economy": "expansion"},
        },
    },
    "prior_marginal": {
        "summary": "Prior marginal P(regime) — no evidence",
        "value": {
            "nodes": [
                {"id": "economy", "states": ["expansion", "recession"]},
                {"id": "regime", "states": ["bull", "bear", "neutral"]},
            ],
            "edges": [{"parent": "economy", "child": "regime"}],
            "priors": [{"node": "economy", "probs": [0.70, 0.30]}],
            "cpts": [
                {
                    "node": "regime",
                    "rows": [
                        {"parents": ["expansion"], "probs": [0.60, 0.10, 0.30]},
                        {"parents": ["recession"], "probs": [0.20, 0.60, 0.20]},
                    ],
                }
            ],
            "query": "regime",
            "evidence": {},
        },
    },
}


def _build_network(req: InferRequest) -> BayesianNetwork:
    """Construct a BayesianNetwork from the request payload."""
    net = BayesianNetwork()
    for node in req.nodes:
        net.add_node(node.id, node.states)
    for edge in req.edges:
        net.add_edge(edge.parent, edge.child)
    for prior in req.priors:
        net.set_prior(prior.node, prior.probs)
    for i, cpt_def in enumerate(req.cpts):
        try:
            _set_cpt(net, cpt_def)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"cpts[{i}] (node '{cpt_def.node}'): {exc}") from exc
    return net


def _set_cpt(net: BayesianNetwork, cpt_def: CptDef) -> None:
    table = {tuple(row.parents): row.probs for row in cpt_def.rows}
    net.set_cpt(cpt_def.node, table)


@router.post("/update", response_model=PosteriorSummary)
def bayesian_update(
    req: Annotated[UpdateRequest, Body(openapi_examples=_BAYESIAN_EXAMPLES)],  # type: ignore[arg-type]
) -> PosteriorSummary:
    """Bayesian conjugate update (beta or normal) given a prior and evidence list."""
    try:
        return update(req.prior, req.evidence)
    except ValueError as e:
        _logger.warning("%s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/networks", response_model=list[NetworkInfo])
def list_named_networks() -> list[NetworkInfo]:
    """Return all pre-built networks registered in the server."""
    return [NetworkInfo(**entry) for entry in list_networks()]


@router.post(
    "/networks/{name}/infer",
    response_model=InferResponse,
    dependencies=[Depends(_require_oracle_key)],
)
def named_network_infer(name: str, req: NamedInferRequest) -> InferResponse:
    """Compute P(query | evidence) on a pre-built named network.

    Use ``GET /networks`` to list available network names.
    """
    net = get_network(name)
    if net is None:
        raise HTTPException(status_code=404, detail=f"Network '{name}' not found")
    try:
        posterior = infer(net, req.query, req.evidence or None)
    except (ValueError, KeyError) as e:
        _logger.warning("%s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    return InferResponse(query=req.query, evidence=req.evidence, posterior=posterior)


@router.post("/infer", response_model=InferResponse, dependencies=[Depends(_require_oracle_key)])
def bayesian_infer(
    req: Annotated[InferRequest, Body(openapi_examples=_INFER_EXAMPLES)],  # type: ignore[arg-type]
) -> InferResponse:
    """Compute P(query | evidence) via Variable Elimination on a discrete Bayesian Network.

    Pass the full network definition (nodes, edges, priors, CPTs) together with
    the query node and any observed evidence.  The response contains the normalized
    posterior distribution over all states of the query node.

    Set `evidence` to `{}` to obtain the prior marginal P(query).
    """
    try:
        net = _build_network(req)
        posterior = infer(net, req.query, req.evidence or None)
    except (ValueError, KeyError) as e:
        _logger.warning("%s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    return InferResponse(query=req.query, evidence=req.evidence, posterior=posterior)
