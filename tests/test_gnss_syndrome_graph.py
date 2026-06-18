"""Tests for SyndromeGraph — append-only consistency-check graph (Syndrome Layer)."""

from __future__ import annotations

import numpy as np
import pytest

from gnss.correspondence_layer import CorrespondenceAssessor
from gnss.syndrome_graph import (
    ConstraintType,
    SyndromeGraph,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(n: int = 6) -> tuple[SyndromeGraph, list[str]]:
    """Return a fresh SyndromeGraph and a list of n satellite IDs."""
    sat_ids = [f"G{i:02d}" for i in range(1, n + 1)]
    return SyndromeGraph(), sat_ids


def _add_nominal(
    graph: SyndromeGraph,
    sat_ids: list[str],
    epoch: int = 0,
    noise_std: float = 0.3,
    rng: np.random.Generator | None = None,
) -> list:
    """Add one nominal epoch (zero-mean Doppler deviations)."""
    if rng is None:
        rng = np.random.default_rng(epoch)
    dop = rng.normal(0.0, noise_std, len(sat_ids))
    assessor = CorrespondenceAssessor()
    corr = assessor.assess_epoch(epoch, sat_ids, dop)
    return graph.add_epoch(
        epoch, sat_ids, list(corr.states), dop, raim_chi2=0.0, raim_threshold=16.92
    )


def _add_spoofing(
    graph: SyndromeGraph,
    sat_ids: list[str],
    epoch: int = 0,
    common_bias: float = 5.0,
    noise_std: float = 0.05,
) -> list:
    """Add one spoofing epoch (coherent common-mode Doppler bias)."""
    rng = np.random.default_rng(epoch)
    dop = np.ones(len(sat_ids)) * common_bias + rng.normal(0.0, noise_std, len(sat_ids))
    assessor = CorrespondenceAssessor()
    corr = assessor.assess_epoch(epoch, sat_ids, dop)
    return graph.add_epoch(
        epoch, sat_ids, list(corr.states), dop, raim_chi2=0.0, raim_threshold=16.92
    )


# ---------------------------------------------------------------------------
# Append-only invariant
# ---------------------------------------------------------------------------


class TestAppendOnlyInvariant:
    def test_empty_graph_has_no_edges(self) -> None:
        graph, _ = _make_graph()
        assert graph.total_edges() == 0

    def test_empty_epoch_query_returns_empty(self) -> None:
        graph, _ = _make_graph()
        assert graph.edges_at_epoch(99) == []
        assert graph.violated_constraints_at(99) == []

    def test_duplicate_epoch_raises(self) -> None:
        graph, sat_ids = _make_graph()
        _add_nominal(graph, sat_ids, epoch=0)
        with pytest.raises(ValueError, match="already processed"):
            _add_nominal(graph, sat_ids, epoch=0)

    def test_all_edges_returns_copy(self) -> None:
        """Mutating the returned list must not affect the graph."""
        graph, sat_ids = _make_graph()
        _add_spoofing(graph, sat_ids, epoch=0)
        before_count = graph.total_edges()
        edges_copy = graph.all_edges()
        edges_copy.clear()
        assert graph.total_edges() == before_count

    def test_edges_grow_monotonically(self) -> None:
        graph, sat_ids = _make_graph()
        prev = 0
        for t in range(5):
            _add_spoofing(graph, sat_ids, epoch=t)
            curr = graph.total_edges()
            assert curr >= prev
            prev = curr


# ---------------------------------------------------------------------------
# Cross-satellite Doppler coherence constraint
# ---------------------------------------------------------------------------


class TestCrossStaDooplerConstraint:
    def test_nominal_no_cross_sat_edge(self) -> None:
        """Zero-mean Doppler deviations should not trigger the coherence constraint."""
        graph, sat_ids = _make_graph(6)
        edges = _add_nominal(graph, sat_ids)
        types = [e.constraint_type for e in edges]
        assert ConstraintType.CROSS_SAT_DOPPLER not in types

    def test_spoofing_triggers_cross_sat_edge(self) -> None:
        """Common-mode bias 5 Hz with noise 0.05 Hz → coherent_snr >> threshold."""
        graph, sat_ids = _make_graph(6)
        edges = _add_spoofing(graph, sat_ids, common_bias=5.0)
        types = [e.constraint_type for e in edges]
        assert ConstraintType.CROSS_SAT_DOPPLER in types

    def test_cross_sat_edge_value_exceeds_threshold(self) -> None:
        graph, sat_ids = _make_graph(6)
        edges = _add_spoofing(graph, sat_ids)
        cs_edges = [e for e in edges if e.constraint_type == ConstraintType.CROSS_SAT_DOPPLER]
        assert len(cs_edges) == 1
        assert cs_edges[0].value > cs_edges[0].threshold
        assert cs_edges[0].is_violated()

    def test_cross_sat_node_satellite_id_is_all(self) -> None:
        """Global constraint must use 'ALL' sentinel satellite ID."""
        graph, sat_ids = _make_graph(4)
        edges = _add_spoofing(graph, sat_ids)
        cs_edges = [e for e in edges if e.constraint_type == ConstraintType.CROSS_SAT_DOPPLER]
        assert cs_edges[0].node_a.satellite_id == "ALL"
        assert cs_edges[0].node_b.satellite_id == "ALL"


# ---------------------------------------------------------------------------
# RAIM geometry constraint
# ---------------------------------------------------------------------------


class TestRAIMConstraint:
    def test_low_chi2_no_raim_edge(self) -> None:
        graph, sat_ids = _make_graph(6)
        assessor = CorrespondenceAssessor()
        dop = np.zeros(6)
        corr = assessor.assess_epoch(0, sat_ids, dop)
        edges = graph.add_epoch(
            0, sat_ids, list(corr.states), dop, raim_chi2=5.0, raim_threshold=16.92
        )
        assert ConstraintType.GEOMETRY_RAIM not in [e.constraint_type for e in edges]

    def test_high_chi2_raim_edge(self) -> None:
        graph, sat_ids = _make_graph(6)
        assessor = CorrespondenceAssessor()
        dop = np.zeros(6)
        corr = assessor.assess_epoch(0, sat_ids, dop)
        edges = graph.add_epoch(
            0, sat_ids, list(corr.states), dop, raim_chi2=30.0, raim_threshold=16.92
        )
        assert ConstraintType.GEOMETRY_RAIM in [e.constraint_type for e in edges]

    def test_raim_edge_uses_constellation_node(self) -> None:
        graph, sat_ids = _make_graph(5)
        assessor = CorrespondenceAssessor()
        dop = np.zeros(5)
        corr = assessor.assess_epoch(0, sat_ids, dop)
        edges = graph.add_epoch(
            0, sat_ids, list(corr.states), dop, raim_chi2=50.0, raim_threshold=16.92
        )
        raim_edges = [e for e in edges if e.constraint_type == ConstraintType.GEOMETRY_RAIM]
        assert raim_edges[0].node_a.satellite_id == "CONSTELLATION"


# ---------------------------------------------------------------------------
# C/N0 coherence constraint
# ---------------------------------------------------------------------------


class TestCN0CoherenceConstraint:
    def test_high_cn0_spread_no_edge(self) -> None:
        """Wide C/N0 spread (nominal: different elevations) → no edge."""
        graph, sat_ids = _make_graph(6)
        assessor = CorrespondenceAssessor()
        dop = np.zeros(6)
        corr = assessor.assess_epoch(0, sat_ids, dop)
        cn0 = np.array([35.0, 38.0, 40.0, 42.0, 44.0, 46.0])  # std/mean ≈ 0.08 > 0.05
        edges = graph.add_epoch(
            0, sat_ids, list(corr.states), dop, raim_chi2=0.0, raim_threshold=16.92, cn0_values=cn0
        )
        assert ConstraintType.CN0_COHERENCE not in [e.constraint_type for e in edges]

    def test_low_cn0_spread_triggers_edge(self) -> None:
        """All sats same C/N0 (single transmitter) → spread collapse → edge added."""
        graph, sat_ids = _make_graph(6)
        assessor = CorrespondenceAssessor()
        dop = np.zeros(6)
        corr = assessor.assess_epoch(0, sat_ids, dop)
        cn0 = np.array([40.01, 40.00, 39.99, 40.01, 40.00, 40.00])  # std/mean ≈ 0.0002
        edges = graph.add_epoch(
            0, sat_ids, list(corr.states), dop, raim_chi2=0.0, raim_threshold=16.92, cn0_values=cn0
        )
        assert ConstraintType.CN0_COHERENCE in [e.constraint_type for e in edges]

    def test_cn0_edge_is_violated(self) -> None:
        """CN0_COHERENCE fires when spread_ratio < threshold (value < threshold)."""
        graph, sat_ids = _make_graph(4)
        assessor = CorrespondenceAssessor()
        dop = np.zeros(4)
        corr = assessor.assess_epoch(0, sat_ids, dop)
        cn0 = np.array([40.0, 40.0, 40.0, 40.0])  # zero std
        edges = graph.add_epoch(
            0, sat_ids, list(corr.states), dop, raim_chi2=0.0, raim_threshold=16.92, cn0_values=cn0
        )
        cn0_edges = [e for e in edges if e.constraint_type == ConstraintType.CN0_COHERENCE]
        if cn0_edges:
            assert cn0_edges[0].is_violated()


# ---------------------------------------------------------------------------
# Temporal phase jump constraint
# ---------------------------------------------------------------------------


class TestTemporalPhaseConstraint:
    def test_smooth_phases_no_edge(self) -> None:
        """Phase increment below threshold → no temporal edge."""
        graph, sat_ids = _make_graph(4)
        assessor = CorrespondenceAssessor()
        dop = np.zeros(4)
        corr = assessor.assess_epoch(1, sat_ids, dop)
        prev_phases = np.zeros(4)
        curr_phases = np.array([0.1, 0.1, 0.1, 0.1])  # small change
        edges = graph.add_epoch(
            1,
            sat_ids,
            list(corr.states),
            dop,
            raim_chi2=0.0,
            raim_threshold=16.92,
            carrier_phases=curr_phases,
            prev_carrier_phases=prev_phases,
        )
        assert ConstraintType.TEMPORAL_PHASE not in [e.constraint_type for e in edges]

    def test_phase_jump_triggers_edge(self) -> None:
        """Phase jump > threshold → temporal edge per affected satellite."""
        graph, sat_ids = _make_graph(4)
        assessor = CorrespondenceAssessor()
        dop = np.zeros(4)
        corr = assessor.assess_epoch(1, sat_ids, dop)
        prev_phases = np.zeros(4)
        curr_phases = np.array([0.1, 1.0, 0.1, 0.1])  # sat G02 jumps by 1.0 cycle
        edges = graph.add_epoch(
            1,
            sat_ids,
            list(corr.states),
            dop,
            raim_chi2=0.0,
            raim_threshold=16.92,
            carrier_phases=curr_phases,
            prev_carrier_phases=prev_phases,
        )
        phase_edges = [e for e in edges if e.constraint_type == ConstraintType.TEMPORAL_PHASE]
        assert len(phase_edges) == 1
        assert phase_edges[0].node_a.satellite_id == "G02"
        assert phase_edges[0].node_b.satellite_id == "G02"


# ---------------------------------------------------------------------------
# Auth mismatch constraint
# ---------------------------------------------------------------------------


class TestAuthMismatchConstraint:
    def test_no_mismatch_when_unknown_auth(self) -> None:
        """Physical GENUINE + crypto UNKNOWN → coherent → no AUTH_MISMATCH edge."""
        graph, sat_ids = _make_graph(4)
        assessor = CorrespondenceAssessor()
        dop = np.zeros(4)
        # No auth flags → UNKNOWN → coherent with GENUINE
        corr = assessor.assess_epoch(0, sat_ids, dop, osnma_auth_per_sat=None)
        edges = graph.add_epoch(
            0, sat_ids, list(corr.states), dop, raim_chi2=0.0, raim_threshold=16.92
        )
        assert ConstraintType.AUTH_MISMATCH not in [e.constraint_type for e in edges]

    def test_mismatch_when_spoofed_and_authenticated(self) -> None:
        """Physical SPOOFED + crypto AUTHENTICATED → incoherent → AUTH_MISMATCH edge."""
        graph, sat_ids = _make_graph(4)
        assessor = CorrespondenceAssessor()
        # High coherent SNR → SPOOFED physical hypothesis.
        dop = np.ones(4) * 8.0 + np.random.default_rng(0).normal(0, 0.01, 4)
        # All sats pass OSNMA auth → AUTHENTICATED.
        auth_flags = [True, True, True, True]
        corr = assessor.assess_epoch(0, sat_ids, dop, osnma_auth_per_sat=auth_flags)
        edges = graph.add_epoch(
            0, sat_ids, list(corr.states), dop, raim_chi2=0.0, raim_threshold=16.92
        )
        mismatch_edges = [e for e in edges if e.constraint_type == ConstraintType.AUTH_MISMATCH]
        assert len(mismatch_edges) == 4  # one per satellite


# ---------------------------------------------------------------------------
# Syndrome score and node fault count
# ---------------------------------------------------------------------------


class TestSyndromeSummary:
    def test_nominal_score_zero(self) -> None:
        graph, sat_ids = _make_graph(6)
        _add_nominal(graph, sat_ids, epoch=0, rng=np.random.default_rng(7))
        assert graph.syndrome_score(0) == 0.0

    def test_score_bounded_by_one(self) -> None:
        graph, sat_ids = _make_graph(6)
        _add_spoofing(graph, sat_ids, epoch=0)
        score = graph.syndrome_score(0)
        assert 0.0 <= score <= 1.0

    def test_node_fault_count_increases_with_spoofing(self) -> None:
        graph, sat_ids = _make_graph(6)
        for t in range(3):
            _add_spoofing(graph, sat_ids, epoch=t)
        # "ALL" sentinel appears in all CROSS_SAT_DOPPLER edges.
        count = graph.node_fault_count("ALL", last_n_epochs=5)
        assert count >= 3

    def test_node_fault_count_zero_for_unaffected_sat(self) -> None:
        graph, sat_ids = _make_graph(6)
        # Only add nominal epochs.
        for t in range(3):
            _add_nominal(graph, sat_ids, epoch=t)
        # G01 was not in any violation.
        assert graph.node_fault_count("G01", last_n_epochs=5) == 0


# ---------------------------------------------------------------------------
# Digest determinism
# ---------------------------------------------------------------------------


class TestSyndromeDigest:
    def test_digest_deterministic_same_input(self) -> None:
        """Identical inputs must produce identical edge digests."""
        dop = np.ones(4) * 5.0
        sat_ids = [f"G{i:02d}" for i in range(1, 5)]
        assessor = CorrespondenceAssessor()

        g1 = SyndromeGraph()
        g2 = SyndromeGraph()
        corr1 = assessor.assess_epoch(0, sat_ids, dop)
        corr2 = assessor.assess_epoch(0, sat_ids, dop)
        edges1 = g1.add_epoch(0, sat_ids, list(corr1.states), dop, 0.0, 16.92)
        edges2 = g2.add_epoch(0, sat_ids, list(corr2.states), dop, 0.0, 16.92)

        assert len(edges1) == len(edges2)
        for e1, e2 in zip(edges1, edges2):
            assert e1.digest == e2.digest

    def test_digest_changes_with_different_value(self) -> None:
        """Different test statistic value must produce different digest."""
        sat_ids = [f"G{i:02d}" for i in range(1, 5)]
        dop_a = np.ones(4) * 5.0
        dop_b = np.ones(4) * 8.0
        assessor = CorrespondenceAssessor()

        g1, g2 = SyndromeGraph(), SyndromeGraph()
        corr_a = assessor.assess_epoch(0, sat_ids, dop_a)
        corr_b = assessor.assess_epoch(0, sat_ids, dop_b)
        edges_a = g1.add_epoch(0, sat_ids, list(corr_a.states), dop_a, 0.0, 16.92)
        edges_b = g2.add_epoch(0, sat_ids, list(corr_b.states), dop_b, 0.0, 16.92)

        digests_a = {e.digest for e in edges_a}
        digests_b = {e.digest for e in edges_b}
        assert digests_a != digests_b
