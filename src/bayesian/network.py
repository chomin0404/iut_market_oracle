"""Discrete Bayesian Network with exact inference via Variable Elimination.

Mathematical basis
------------------
Joint distribution (chain rule of probability):

    P(X₁, X₂, ..., Xₙ) = ∏ᵢ P(Xᵢ | pa(Xᵢ))

where pa(Xᵢ) is the parent set of Xᵢ in the DAG.
Root nodes (no parents) have unconditional priors P(Xᵢ).

Posterior query:

    P(Q | E=e) ∝ Σ_{hidden} ∏ᵢ P(Xᵢ | pa(Xᵢ))

Variable Elimination (VE) algorithm
------------------------------------
1. Initialize one factor φᵢ = P(Xᵢ | pa(Xᵢ)) per node.
2. Restrict every factor containing evidence variable Eⱼ to Eⱼ = eⱼ.
3. For each hidden variable H (non-query, non-evidence):
   a. Collect all factors that mention H.
   b. Multiply them into a joint factor φ*.
   c. Sum out H: φ'(rest) = Σ_H φ*(H, rest).
4. Multiply all remaining factors.
5. If any non-query variable survives, sum it out.
6. Normalize the query factor to obtain P(Q | E=e).

Complexity: exponential in the induced tree-width.
For the small networks (≤ 15 nodes, ≤ 5 states) typical in this research
the Cartesian-product factor multiplication is fast enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as _iproduct

import numpy as np

# ---------------------------------------------------------------------------
# Internal factor representation
# ---------------------------------------------------------------------------


@dataclass
class _Factor:
    """Factor φ(V₁, V₂, ..., Vₖ) — a function over joint discrete states.

    variables: ordered list of variable names in this factor.
    values:    numpy array of shape (|V₁|, |V₂|, ..., |Vₖ|).
               Empty variables means a scalar factor; values.shape == ().
    """

    variables: list[str]
    values: np.ndarray

    def restrict(self, var: str, state_idx: int) -> _Factor:
        """Return a new factor with *var* fixed to *state_idx*."""
        axis = self.variables.index(var)
        idx: list[int | slice] = [slice(None)] * len(self.variables)
        idx[axis] = state_idx
        return _Factor(
            variables=[v for v in self.variables if v != var],
            values=self.values[tuple(idx)],
        )

    def marginalize(self, var: str) -> _Factor:
        """Return a new factor with *var* summed out."""
        axis = self.variables.index(var)
        return _Factor(
            variables=[v for v in self.variables if v != var],
            values=self.values.sum(axis=axis),
        )

    def normalize(self) -> _Factor:
        """Return factor scaled so values sum to 1."""
        total = float(self.values.sum())
        if total <= 0.0:
            raise ValueError(
                "Cannot normalize a zero factor — evidence may be inconsistent "
                "with the network (zero-probability state combination)."
            )
        return _Factor(variables=list(self.variables), values=self.values / total)


def _factor_product(f1: _Factor, f2: _Factor) -> _Factor:
    """Return φ₁ × φ₂ over the union of their variables.

    Handles scalar (empty-variable) factors as a special case.
    Iterates over the Cartesian product of states for correctness and clarity.
    """
    # Scalar cases
    if not f1.variables:
        return _Factor(variables=list(f2.variables), values=float(f1.values) * f2.values)
    if not f2.variables:
        return _Factor(variables=list(f1.variables), values=f1.values * float(f2.values))

    union_vars: list[str] = list(f1.variables)
    for v in f2.variables:
        if v not in union_vars:
            union_vars.append(v)

    state_counts: dict[str, int] = {
        **dict(zip(f1.variables, f1.values.shape)),
        **dict(zip(f2.variables, f2.values.shape)),
    }
    shape = tuple(state_counts[v] for v in union_vars)
    result = np.empty(shape, dtype=float)

    for joint in _iproduct(*(range(state_counts[v]) for v in union_vars)):
        state = dict(zip(union_vars, joint))
        i1 = tuple(state[v] for v in f1.variables)
        i2 = tuple(state[v] for v in f2.variables)
        result[joint] = f1.values[i1] * f2.values[i2]

    return _Factor(variables=union_vars, values=result)


# ---------------------------------------------------------------------------
# Node specification (internal)
# ---------------------------------------------------------------------------


@dataclass
class _NodeSpec:
    node_id: str
    states: list[str]
    cpt: np.ndarray | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class BayesianNetwork:
    """Discrete Bayesian Network with Variable Elimination inference.

    Workflow
    --------
    1. Register nodes with :meth:`add_node`.
    2. Define the DAG structure with :meth:`add_edge`.
    3. Assign priors (:meth:`set_prior`) and CPTs (:meth:`set_cpt`).
    4. Inject evidence with :meth:`observe`.
    5. Query posteriors with :meth:`posterior` or :meth:`update`.

    Example (market regime model)
    ------------------------------
    >>> net = BayesianNetwork()
    >>> net.add_node("economy", states=["expansion", "recession"])
    >>> net.add_node("regime",  states=["bull", "bear", "neutral"])
    >>> net.add_edge("economy", "regime")
    >>> net.set_prior("economy", [0.70, 0.30])
    >>> net.set_cpt("regime", {
    ...     ("expansion",): [0.60, 0.10, 0.30],
    ...     ("recession",): [0.20, 0.60, 0.20],
    ... })
    >>> net.observe("economy", "expansion")
    >>> net.posterior("regime")
    {'bull': 0.6, 'bear': 0.1, 'neutral': 0.3}
    """

    def __init__(self) -> None:
        self._specs: dict[str, _NodeSpec] = {}
        self._parents: dict[str, list[str]] = {}  # child  → [parent, ...]
        self._children: dict[str, list[str]] = {}  # parent → [child, ...]
        self._evidence: dict[str, int] = {}  # node_id → state index

    # ------------------------------------------------------------------
    # Structure definition
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, states: list[str]) -> None:
        """Register a discrete random variable with the given state labels.

        Parameters
        ----------
        node_id:
            Unique identifier for this node.
        states:
            Ordered list of mutually exclusive state labels (≥ 2 required).
        """
        if node_id in self._specs:
            raise ValueError(f"Node '{node_id}' already exists in the network")
        if len(states) < 2:
            raise ValueError(f"Node '{node_id}' must have at least 2 states, got {len(states)}")
        if len(set(states)) != len(states):
            raise ValueError(f"Node '{node_id}' has duplicate state labels: {states}")
        self._specs[node_id] = _NodeSpec(node_id=node_id, states=list(states))
        self._parents[node_id] = []
        self._children[node_id] = []

    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed edge parent → child to the DAG.

        Raises
        ------
        ValueError
            If either node is unknown, a self-loop is attempted, the edge
            already exists, or adding the edge would create a cycle.
        """
        for n in (parent, child):
            if n not in self._specs:
                raise ValueError(f"Node '{n}' not found; call add_node() first")
        if parent == child:
            raise ValueError(f"Self-loop not allowed: '{parent}' → '{child}'")
        if parent in self._parents[child]:
            raise ValueError(f"Edge '{parent}' → '{child}' already exists")
        self._parents[child].append(parent)
        self._children[parent].append(child)
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        """Kahn's algorithm: raise ValueError if the graph contains a cycle."""
        in_deg = {n: len(p) for n, p in self._parents.items()}
        queue = [n for n, d in in_deg.items() if d == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for child in self._children[node]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        if visited != len(self._specs):
            raise ValueError("Cycle detected — Bayesian Network must be a DAG")

    # ------------------------------------------------------------------
    # CPT assignment
    # ------------------------------------------------------------------

    def set_prior(self, node_id: str, probs: list[float]) -> None:
        """Set the unconditional prior P(node_id) for a root node.

        Parameters
        ----------
        node_id:
            Must be a root node (no parents).
        probs:
            Probability mass for each state; must sum to 1.0 (tolerance 1e-6).
        """
        spec = self._require_node(node_id)
        if self._parents[node_id]:
            raise ValueError(
                f"Node '{node_id}' has parents {self._parents[node_id]}; "
                "use set_cpt() for conditional nodes"
            )
        arr = np.array(probs, dtype=float)
        if arr.shape != (len(spec.states),):
            raise ValueError(
                f"Prior for '{node_id}': expected {len(spec.states)} values, got {len(arr)}"
            )
        _validate_prob_row(arr, node_id, "prior")
        spec.cpt = arr

    def set_cpt(
        self,
        node_id: str,
        table: dict[tuple[str, ...], list[float]],
    ) -> None:
        """Set the conditional probability table P(node_id | parents).

        Parameters
        ----------
        node_id:
            Target node (must have at least one parent already added via add_edge).
        table:
            Mapping  (parent_state_1, parent_state_2, ...) → [P(X=s₀), P(X=s₁), ...].
            The tuple key order must match the order parents were registered
            via :meth:`add_edge`.  Every combination of parent states must be present.

        Example
        -------
        For node ``regime`` with one parent ``economy`` having states
        ``["expansion", "recession"]``::

            net.set_cpt("regime", {
                ("expansion",): [0.60, 0.10, 0.30],
                ("recession",): [0.20, 0.60, 0.20],
            })
        """
        spec = self._require_node(node_id)
        parents = self._parents[node_id]
        if not parents:
            raise ValueError(f"Node '{node_id}' has no parents; use set_prior() for root nodes")

        parent_state_lists: list[list[str]] = [self._specs[p].states for p in parents]
        cpt_shape = tuple(len(ps) for ps in parent_state_lists) + (len(spec.states),)
        cpt = np.zeros(cpt_shape, dtype=float)

        state_index: list[dict[str, int]] = [
            {s: i for i, s in enumerate(ps)} for ps in parent_state_lists
        ]

        for combo, probs in table.items():
            if len(combo) != len(parents):
                raise ValueError(
                    f"CPT key {combo} has {len(combo)} elements but "
                    f"'{node_id}' has {len(parents)} parent(s)"
                )
            try:
                idx = tuple(state_index[i][s] for i, s in enumerate(combo))
            except KeyError as exc:
                raise ValueError(f"Unknown parent state {exc} in CPT for '{node_id}'") from exc

            row = np.array(probs, dtype=float)
            if row.shape != (len(spec.states),):
                raise ValueError(
                    f"CPT row {combo} for '{node_id}': expected {len(spec.states)} "
                    f"values, got {len(row)}"
                )
            _validate_prob_row(row, node_id, str(combo))
            cpt[idx] = row

        # Verify all parent combinations are covered
        expected_rows = 1
        for ps in parent_state_lists:
            expected_rows *= len(ps)
        if len(table) != expected_rows:
            raise ValueError(
                f"CPT for '{node_id}' has {len(table)} rows but "
                f"{expected_rows} parent-state combinations are required"
            )

        spec.cpt = cpt

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    def observe(self, node_id: str, state: str) -> None:
        """Record evidence: node_id is observed to be in *state*.

        Call :meth:`reset_evidence` to retract all evidence.
        """
        spec = self._require_node(node_id)
        if state not in spec.states:
            raise ValueError(
                f"State '{state}' not valid for node '{node_id}'. Valid states: {spec.states}"
            )
        self._evidence[node_id] = spec.states.index(state)

    def reset_evidence(self) -> None:
        """Retract all evidence; next call to :meth:`posterior` uses only priors."""
        self._evidence.clear()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def posterior(self, query: str) -> dict[str, float]:
        """Compute P(query | current evidence) via Variable Elimination.

        Parameters
        ----------
        query:
            Node ID to query.

        Returns
        -------
        dict[str, float]
            Mapping state → posterior probability.  Values sum to 1.0.

        Raises
        ------
        ValueError
            If any CPT is missing, or evidence is inconsistent (zero probability).
        """
        self._require_node(query)

        # Degenerate case: query is directly observed
        if query in self._evidence:
            ev_state = self._specs[query].states[self._evidence[query]]
            return {s: (1.0 if s == ev_state else 0.0) for s in self._specs[query].states}

        # Validate all CPTs are assigned
        for nid, spec in self._specs.items():
            if spec.cpt is None:
                raise ValueError(
                    f"CPT not set for node '{nid}'. "
                    "Call set_prior() or set_cpt() before running inference."
                )

        # Step 1: Build initial factors P(Xᵢ | pa(Xᵢ))
        factors: list[_Factor] = []
        for nid, spec in self._specs.items():
            variables = self._parents[nid] + [nid]
            assert spec.cpt is not None  # validated above
            factors.append(_Factor(variables=variables, values=spec.cpt.copy()))

        # Step 2: Restrict factors to evidence
        for ev_node, ev_idx in self._evidence.items():
            new_factors: list[_Factor] = []
            for f in factors:
                if ev_node in f.variables:
                    f = f.restrict(ev_node, ev_idx)
                new_factors.append(f)
            factors = new_factors

        # Step 3: Eliminate hidden variables
        hidden = set(self._specs.keys()) - set(self._evidence.keys()) - {query}
        for h in hidden:
            involved = [f for f in factors if h in f.variables]
            rest = [f for f in factors if h not in f.variables]
            if not involved:
                continue
            joint = involved[0]
            for f in involved[1:]:
                joint = _factor_product(joint, f)
            joint = joint.marginalize(h)
            factors = rest + [joint]

        # Step 4: Multiply remaining factors
        result = factors[0]
        for f in factors[1:]:
            result = _factor_product(result, f)

        # Step 5: Sum out any surviving non-query variables
        for v in list(result.variables):
            if v != query:
                result = result.marginalize(v)

        if query not in result.variables:
            raise RuntimeError(
                f"Variable Elimination failed: query variable '{query}' not in result factor. "
                "This indicates a network topology issue."
            )

        # Step 6: Normalize
        result = result.normalize()

        spec = self._specs[query]
        q_axis = result.variables.index(query)
        probs = (
            result.values
            if result.values.ndim == 1
            else result.values.take(range(len(spec.states)), axis=q_axis)
        )
        return {state: float(probs[i]) for i, state in enumerate(spec.states)}

    def update(
        self,
        evidence: dict[str, str],
        queries: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute posteriors for multiple queries given a fresh evidence set.

        Existing evidence is saved and restored after this call.

        Parameters
        ----------
        evidence:
            Mapping node_id → observed state.
        queries:
            Node IDs to compute posteriors for.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping query_node_id → {state → posterior probability}.
        """
        saved = dict(self._evidence)
        try:
            for nid, state in evidence.items():
                self.observe(nid, state)
            return {q: self.posterior(q) for q in queries}
        finally:
            self._evidence = saved

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def topological_order(self) -> list[str]:
        """Return all node IDs in topological order (parents before children)."""
        in_deg = {n: len(p) for n, p in self._parents.items()}
        queue = sorted(n for n, d in in_deg.items() if d == 0)
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in sorted(self._children[node]):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        return order

    def node_states(self, node_id: str) -> list[str]:
        """Return the state labels for *node_id*."""
        return list(self._require_node(node_id).states)

    def _require_node(self, node_id: str) -> _NodeSpec:
        if node_id not in self._specs:
            raise ValueError(f"Node '{node_id}' not found in network")
        return self._specs[node_id]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _validate_prob_row(arr: np.ndarray, node_id: str, context: str) -> None:
    """Raise ValueError if *arr* is not a valid probability distribution."""
    if np.any(arr < 0.0):
        raise ValueError(f"Probabilities for '{node_id}' ({context}) must be non-negative")
    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Probabilities for '{node_id}' ({context}) must sum to 1.0, got {total:.8f}"
        )
