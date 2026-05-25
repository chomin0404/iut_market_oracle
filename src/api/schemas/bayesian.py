"""Request schemas for the Bayesian update router."""

from __future__ import annotations

import secrets
from typing import Annotated, Literal

import numpy as np
import scipy.linalg as la
from pydantic import BaseModel, Field, field_validator, model_validator

from bayesian.network import BayesianNetwork
from schemas import Evidence, PriorSpec


class UpdateRequest(BaseModel):
    prior: PriorSpec
    evidence: list[Evidence]


# ---------------------------------------------------------------------------
# MCMC target specs
# ---------------------------------------------------------------------------


_MAX_DIM = 100  # guard against huge allocation


class NormalTargetSpec(BaseModel):
    """Isotropic multivariate Gaussian N(mu, sigma^2 I)."""

    type: Literal["normal"] = "normal"
    mu: list[float] = Field(
        ..., min_length=1, max_length=_MAX_DIM, description="Mean vector, length = dim"
    )
    sigma: float = Field(1.0, gt=0.0, description="Isotropic standard deviation")


class MultivariateNormalTargetSpec(BaseModel):
    """Full-covariance Gaussian N(mu, Sigma)."""

    type: Literal["multivariate_normal"] = "multivariate_normal"
    mu: list[float] = Field(
        ..., min_length=1, max_length=_MAX_DIM, description="Mean vector, length = dim"
    )
    cov: list[list[float]] = Field(..., description="Covariance matrix, shape (dim, dim)")

    @field_validator("cov")
    @classmethod
    def _check_square_and_symmetric(cls, v: list[list[float]]) -> list[list[float]]:
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("cov must be a square matrix")
        if not np.allclose(arr, arr.T, atol=1e-8):
            raise ValueError("cov must be symmetric")
        return v

    @model_validator(mode="after")
    def _check_dim_and_pd(self) -> MultivariateNormalTargetSpec:
        arr = np.asarray(self.cov, dtype=float)
        if arr.shape[0] != len(self.mu):
            raise ValueError(
                f"cov dim {arr.shape[0]} does not match mu length {len(self.mu)}"
            )
        try:
            la.cholesky(arr, lower=True)
        except la.LinAlgError:
            raise ValueError("cov must be positive definite")
        return self


# Discriminated union — dispatch on the `type` field.
TargetSpec = Annotated[
    NormalTargetSpec | MultivariateNormalTargetSpec,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# MCMC shared response
# ---------------------------------------------------------------------------


class TraceSummary(BaseModel):
    """Per-dimension descriptive statistics of the MCMC chain."""

    mean: list[float] = Field(..., description="Per-dimension posterior mean")
    std: list[float] = Field(..., description="Per-dimension posterior standard deviation")
    q2_5: list[float] = Field(..., description="2.5th percentile per dimension")
    q25: list[float] = Field(..., description="25th percentile per dimension")
    q50: list[float] = Field(..., description="Median per dimension")
    q75: list[float] = Field(..., description="75th percentile per dimension")
    q97_5: list[float] = Field(..., description="97.5th percentile per dimension")


class ConvergenceDiagnostics(BaseModel):
    """MCMC convergence diagnostics."""

    ess: list[float] = Field(
        ...,
        description=(
            "Effective Sample Size per dimension, estimated via the monotone "
            "positive-sequence estimator (Geyer 1992). Values ≤ n_samples."
        ),
    )
    r_hat: list[float] = Field(
        ...,
        description=(
            "Split-R-hat per dimension (Vehtari et al. 2021). "
            "Values close to 1.0 indicate convergence; R-hat < 1.01 is the common threshold."
        ),
    )
    trace_summary: TraceSummary = Field(
        ...,
        description="Per-dimension descriptive statistics of the post-burn-in chain.",
    )


class MCMCSamplesResponse(BaseModel):
    """Collected chain and acceptance diagnostics."""

    samples: list[list[float]] = Field(..., description="shape (n_samples, dim)")
    acceptance_rate: float
    n_accepted: int
    n_total: int
    diagnostics: ConvergenceDiagnostics


# ---------------------------------------------------------------------------
# MH
# ---------------------------------------------------------------------------


def _random_seed() -> int:
    return secrets.randbits(32)


class MHRequest(BaseModel):
    target: TargetSpec
    step_size: float = Field(0.5, gt=0.0, description="Gaussian RW step size sigma")
    initial: list[float] = Field(
        ..., min_length=1, max_length=_MAX_DIM, description="Starting state, length = dim"
    )
    n_samples: int = Field(..., ge=1, le=10_000)
    seed: int = Field(
        default_factory=_random_seed, ge=0, description="RNG seed. Omit for random."
    )
    burn_in: int = Field(0, ge=0)
    thin: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _check_initial_dim(self) -> MHRequest:
        if len(self.initial) != len(self.target.mu):
            raise ValueError(
                f"initial length {len(self.initial)} must match "
                f"target mu length {len(self.target.mu)}"
            )
        return self


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------


class HMCRequest(BaseModel):
    target: TargetSpec
    step_size: float = Field(..., gt=0.0, description="Leapfrog step size epsilon")
    n_leapfrog: int = Field(
        10,
        ge=1,
        le=1000,
        description=(
            "Number of leapfrog integration steps L per proposal. "
            "Larger L explores farther per step but costs more compute. "
            "Typical range: 5–50. Must be ≥ 1."
        ),
    )
    initial: list[float] = Field(
        ..., min_length=1, max_length=_MAX_DIM, description="Starting state, length = dim"
    )
    n_samples: int = Field(..., ge=1, le=10_000)
    seed: int = Field(
        default_factory=_random_seed, ge=0, description="RNG seed. Omit for random."
    )
    burn_in: int = Field(0, ge=0)
    thin: int = Field(1, ge=1)
    mass: list[float] | None = Field(
        None,
        max_length=_MAX_DIM,
        description="Diagonal mass matrix, length = dim. Defaults to identity.",
    )

    @model_validator(mode="after")
    def _check_dims(self) -> HMCRequest:
        dim = len(self.target.mu)
        if len(self.initial) != dim:
            raise ValueError(
                f"initial length {len(self.initial)} must match target mu length {dim}"
            )
        if self.mass is not None and len(self.mass) != dim:
            raise ValueError(
                f"mass length {len(self.mass)} must match target mu length {dim}"
            )
        return self


# ---------------------------------------------------------------------------
# Bayesian Network data models
# ---------------------------------------------------------------------------

_MAX_STATES = 50  # max discrete states per node
_MAX_NODES = 200  # max nodes per network spec
_NODE_ID_RE = r"^[A-Za-z_][A-Za-z0-9_\-]*$"


class BNNode(BaseModel):
    """Discrete random variable (node) in a Bayesian Network.

    Invariants
    ----------
    * ``len(states) >= 2`` — at least two mutually exclusive outcomes.
    * All state labels are unique non-empty strings.
    """

    node_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=_NODE_ID_RE,
        description=(
            "Unique node identifier. "
            "Must start with a letter or underscore; may contain alphanumerics, underscores, hyphens."
        ),
    )
    states: list[str] = Field(
        ...,
        min_length=2,
        max_length=_MAX_STATES,
        description="Ordered, mutually exclusive state labels (≥ 2).",
    )

    @field_validator("states")
    @classmethod
    def _unique_nonempty_states(cls, v: list[str]) -> list[str]:
        if any(len(s) == 0 for s in v):
            raise ValueError("state labels must be non-empty strings")
        if len(set(v)) != len(v):
            raise ValueError(f"state labels must be unique, got duplicates in {v}")
        return v


class BNEdge(BaseModel):
    """Directed edge parent → child in the DAG.

    The pair ``(parent, child)`` must be unique across all edges in a network,
    and adding it must not create a cycle — enforced at the network level.
    """

    parent: str = Field(..., description="Parent node ID.")
    child: str = Field(..., description="Child node ID.")

    @model_validator(mode="after")
    def _no_self_loop(self) -> BNEdge:
        if self.parent == self.child:
            raise ValueError(f"Self-loop not allowed: '{self.parent}' → '{self.child}'")
        return self


class CPDRow(BaseModel):
    """One row in a Conditional Probability Table (CPT).

    Each row corresponds to one combination of parent states and specifies
    the conditional distribution P(node | parent_states).

    Invariants
    ----------
    * ``probs[i] >= 0`` for all i.
    * ``sum(probs) == 1.0``  (tolerance 1e-6).
    """

    parent_states: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered parent state labels. "
            "Order must match the sequence of parent nodes in the network's BNEdge list."
        ),
    )
    probs: list[float] = Field(
        ...,
        min_length=2,
        description=(
            "P(node=s₀), P(node=s₁), ... for each state of the target node. "
            "Must be non-negative and sum to 1.0."
        ),
    )

    @field_validator("probs")
    @classmethod
    def _valid_probs(cls, v: list[float]) -> list[float]:
        arr = np.asarray(v, dtype=float)
        if np.any(arr < 0.0):
            raise ValueError("probs must be non-negative")
        total = float(arr.sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"probs must sum to 1.0, got {total:.8f}")
        return v


class CPD(BaseModel):
    """Conditional Probability Distribution for one node.

    Use ``probs`` for root nodes (no parents) and ``rows`` for conditional
    nodes (one row per combination of parent states).  Exactly one of the two
    fields must be supplied.

    Invariants
    ----------
    * Exactly one of ``probs`` / ``rows`` is non-null.
    * ``probs``: non-negative, sums to 1.0.
    * ``rows``: each row satisfies the same probability constraints.
    """

    node_id: str = Field(..., description="Target node this CPD belongs to.")
    probs: list[float] | None = Field(
        None,
        min_length=2,
        description=(
            "Unconditional prior P(node) for a root node (no parents). "
            "Must be non-negative and sum to 1.0."
        ),
    )
    rows: list[CPDRow] | None = Field(
        None,
        min_length=1,
        description=(
            "CPT rows for a conditional node. "
            "One CPDRow per combination of parent states; "
            "all parent-state combinations must be present."
        ),
    )

    @field_validator("probs")
    @classmethod
    def _valid_prior(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        arr = np.asarray(v, dtype=float)
        if np.any(arr < 0.0):
            raise ValueError("probs must be non-negative")
        total = float(arr.sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"probs must sum to 1.0, got {total:.8f}")
        return v

    @model_validator(mode="after")
    def _exactly_one_of_probs_or_rows(self) -> CPD:
        if (self.probs is None) == (self.rows is None):
            raise ValueError(
                "Exactly one of 'probs' (root node) or 'rows' (conditional node) must be provided"
            )
        return self


class BNNetworkSpec(BaseModel):
    """Complete specification of a discrete Bayesian Network.

    Contains nodes, edges (DAG), and CPDs — enough to fully construct a
    ``BayesianNetwork`` instance.

    Cross-field invariants (enforced on construction)
    -------------------------------------------------
    * ``node_id`` values are unique across all nodes.
    * Every edge endpoint references a declared node.
    * No duplicate edges.
    * The edge set is acyclic (DAG).
    * Every node has exactly one CPD.
    * Root nodes (no incoming edges) use ``CPD.probs``; others use ``CPD.rows``.
    * ``CPD.probs`` length matches the node's state count.
    * ``CPDRow`` count equals the product of parent state counts.
    * ``CPDRow.parent_states`` length matches the node's parent count.
    * ``CPDRow.probs`` length matches the node's state count.
    """

    nodes: list[BNNode] = Field(
        ..., min_length=1, max_length=_MAX_NODES, description="All nodes in the network."
    )
    edges: list[BNEdge] = Field(
        default_factory=list, description="Directed edges (parent → child). Empty for root-only networks."
    )
    cpds: list[CPD] = Field(
        ..., min_length=1, description="One CPD per node, covering every node declared in `nodes`."
    )

    @model_validator(mode="after")
    def _validate_network(self) -> BNNetworkSpec:
        # ── 1. Unique node IDs ───────────────────────────────────────────────
        node_map: dict[str, BNNode] = {}
        for node in self.nodes:
            if node.node_id in node_map:
                raise ValueError(f"Duplicate node_id: '{node.node_id}'")
            node_map[node.node_id] = node

        # ── 2. Edges: endpoints exist, no duplicates ─────────────────────────
        parents: dict[str, list[str]] = {nid: [] for nid in node_map}
        children: dict[str, list[str]] = {nid: [] for nid in node_map}
        seen_edges: set[tuple[str, str]] = set()
        for edge in self.edges:
            for endpoint in (edge.parent, edge.child):
                if endpoint not in node_map:
                    raise ValueError(f"Edge references unknown node '{endpoint}'")
            pair = (edge.parent, edge.child)
            if pair in seen_edges:
                raise ValueError(f"Duplicate edge: '{edge.parent}' → '{edge.child}'")
            seen_edges.add(pair)
            parents[edge.child].append(edge.parent)
            children[edge.parent].append(edge.child)

        # ── 3. DAG check (Kahn's algorithm) ─────────────────────────────────
        in_deg = {n: len(p) for n, p in parents.items()}
        queue = [n for n, d in in_deg.items() if d == 0]
        visited = 0
        while queue:
            nid = queue.pop()
            visited += 1
            for child in children[nid]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        if visited != len(node_map):
            raise ValueError("Edge set contains a cycle; network must be a DAG")

        # ── 4. CPDs: unique, reference existing nodes, every node covered ────
        cpd_map: dict[str, CPD] = {}
        for cpd in self.cpds:
            if cpd.node_id not in node_map:
                raise ValueError(f"CPD references unknown node '{cpd.node_id}'")
            if cpd.node_id in cpd_map:
                raise ValueError(f"Duplicate CPD for node '{cpd.node_id}'")
            cpd_map[cpd.node_id] = cpd
        missing = set(node_map) - set(cpd_map)
        if missing:
            raise ValueError(f"Missing CPD for node(s): {sorted(missing)}")

        # ── 5. CPD structure matches node topology ───────────────────────────
        for nid, cpd in cpd_map.items():
            node = node_map[nid]
            node_parents = parents[nid]
            n_states = len(node.states)

            if not node_parents:
                # Root node → must use probs
                if cpd.probs is None:
                    raise ValueError(
                        f"Node '{nid}' has no parents; CPD must use 'probs', not 'rows'"
                    )
                if len(cpd.probs) != n_states:
                    raise ValueError(
                        f"CPD.probs for '{nid}' has {len(cpd.probs)} values "
                        f"but node has {n_states} states"
                    )
            else:
                # Conditional node → must use rows
                if cpd.rows is None:
                    raise ValueError(
                        f"Node '{nid}' has parents {node_parents}; "
                        "CPD must use 'rows', not 'probs'"
                    )
                expected_rows = 1
                for pid in node_parents:
                    expected_rows *= len(node_map[pid].states)
                if len(cpd.rows) != expected_rows:
                    raise ValueError(
                        f"CPD for '{nid}' has {len(cpd.rows)} row(s) "
                        f"but {expected_rows} parent-state combination(s) are required"
                    )
                for row in cpd.rows:
                    if len(row.parent_states) != len(node_parents):
                        raise ValueError(
                            f"CPDRow.parent_states for '{nid}' has "
                            f"{len(row.parent_states)} value(s) "
                            f"but node has {len(node_parents)} parent(s)"
                        )
                    if len(row.probs) != n_states:
                        raise ValueError(
                            f"CPDRow.probs for '{nid}' "
                            f"(parent_states={row.parent_states}) has {len(row.probs)} "
                            f"values but node has {n_states} states"
                        )

        return self


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_network(spec: BNNetworkSpec) -> BayesianNetwork:
    """Construct a :class:`BayesianNetwork` from a validated :class:`BNNetworkSpec`.

    Parameters
    ----------
    spec:
        A fully-validated ``BNNetworkSpec``.  All cross-field invariants are
        guaranteed by Pydantic construction, so no additional checks are
        performed here.

    Returns
    -------
    BayesianNetwork
        Network with all nodes, edges, and CPTs assigned — ready for inference.

    Example
    -------
    >>> spec = BNNetworkSpec(
    ...     nodes=[
    ...         BNNode(node_id="economy", states=["expansion", "recession"]),
    ...         BNNode(node_id="regime",  states=["bull", "bear", "neutral"]),
    ...     ],
    ...     edges=[BNEdge(parent="economy", child="regime")],
    ...     cpds=[
    ...         CPD(node_id="economy", probs=[0.7, 0.3]),
    ...         CPD(node_id="regime", rows=[
    ...             CPDRow(parent_states=["expansion"], probs=[0.6, 0.1, 0.3]),
    ...             CPDRow(parent_states=["recession"], probs=[0.2, 0.6, 0.2]),
    ...         ]),
    ...     ],
    ... )
    >>> net = build_network(spec)
    >>> net.posterior("regime")
    {'bull': ..., 'bear': ..., 'neutral': ...}
    """
    net = BayesianNetwork()

    for node in spec.nodes:
        net.add_node(node.node_id, node.states)

    for edge in spec.edges:
        net.add_edge(edge.parent, edge.child)

    for cpd in spec.cpds:
        if cpd.probs is not None:
            net.set_prior(cpd.node_id, cpd.probs)
        else:
            assert cpd.rows is not None  # guaranteed by BNNetworkSpec validation
            table: dict[tuple[str, ...], list[float]] = {
                tuple(row.parent_states): row.probs for row in cpd.rows
            }
            net.set_cpt(cpd.node_id, table)

    return net


# ---------------------------------------------------------------------------
# Network inference request / response
# ---------------------------------------------------------------------------


class BNInferenceRequest(BaseModel):
    """Request body for POST /bayesian/network/infer."""

    network: BNNetworkSpec = Field(..., description="Fully-specified Bayesian Network.")
    evidence: dict[str, str] = Field(
        default_factory=dict,
        description="Observed states: {node_id: state_label}. Empty means no observations.",
    )
    queries: list[str] = Field(
        ...,
        min_length=1,
        description="Node IDs whose posteriors are requested.",
    )


class BNInferenceResponse(BaseModel):
    """Response body for POST /bayesian/network/infer."""

    posteriors: dict[str, dict[str, float]] = Field(
        ...,
        description=(
            "Posterior distributions: {node_id: {state_label: probability}}. "
            "Each inner dict sums to 1.0."
        ),
    )
