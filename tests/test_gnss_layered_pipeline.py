"""Integration tests for the 5-layer GNSS defense pipeline (LayeredPipeline)."""

from __future__ import annotations

import numpy as np
import pytest

from gnss.certificate import AlarmCertificate, build_certificate
from gnss.decoder import DecisionClass, DecoderResult
from gnss.layered_pipeline import LayeredPipeline
from gnss.syndrome_graph import ConstraintType, SyndromeEdge, SyndromeNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAT_IDS = [f"G{i:02d}" for i in range(1, 7)]
N_SATS = len(SAT_IDS)


def _nominal_doppler(rng: np.random.Generator, noise_std: float = 0.3) -> np.ndarray:
    return rng.normal(0.0, noise_std, N_SATS)


def _spoofing_doppler(
    rng: np.random.Generator, common_bias: float = 5.0, noise_std: float = 0.05
) -> np.ndarray:
    return np.ones(N_SATS) * common_bias + rng.normal(0.0, noise_std, N_SATS)


def _hardware_fault_doppler(
    rng: np.random.Generator, fault_sat_idx: int = 0, bias: float = 20.0
) -> np.ndarray:
    dop = rng.normal(0.0, 0.3, N_SATS)
    dop[fault_sat_idx] += bias
    return dop


def _make_pipeline(run_id: str = "test0000") -> LayeredPipeline:
    return LayeredPipeline(SAT_IDS, run_id=run_id)


# ---------------------------------------------------------------------------
# Nominal scenario
# ---------------------------------------------------------------------------


class TestNominalScenario:
    def test_nominal_decision_is_nominal(self) -> None:
        rng = np.random.default_rng(42)
        pipeline = _make_pipeline("nomtest1")
        result = pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=_nominal_doppler(rng))
        assert result.decoder.decision == DecisionClass.NOMINAL

    def test_nominal_no_syndrome_edges(self) -> None:
        rng = np.random.default_rng(42)
        pipeline = _make_pipeline("nomtest2")
        result = pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=_nominal_doppler(rng))
        assert result.n_syndrome_edges == 0

    def test_nominal_certificate_valid(self) -> None:
        rng = np.random.default_rng(42)
        pipeline = _make_pipeline("nomtest3")
        result = pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=_nominal_doppler(rng))
        cert = result.certificate
        assert isinstance(cert, AlarmCertificate)
        assert cert.is_valid()
        assert cert.decision == DecisionClass.NOMINAL

    def test_nominal_all_invariants_satisfied(self) -> None:
        rng = np.random.default_rng(42)
        pipeline = _make_pipeline("nomtest4")
        result = pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=_nominal_doppler(rng))
        cert = result.certificate
        assert len(cert.checks_failed) == 0
        assert len(cert.invariants_satisfied) > 0

    def test_nominal_anomaly_beliefs_low(self) -> None:
        rng = np.random.default_rng(42)
        pipeline = _make_pipeline("nomtest5")
        result = pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=_nominal_doppler(rng))
        # All beliefs should be below 0.5 (prior-driven).
        for belief in result.decoder.anomaly_beliefs:
            assert belief < 0.5


# ---------------------------------------------------------------------------
# Spoofing scenario
# ---------------------------------------------------------------------------


class TestSpoofingScenario:
    def test_spoofing_decision_is_not_nominal(self) -> None:
        rng = np.random.default_rng(0)
        pipeline = _make_pipeline("spftest1")
        result = pipeline.process_epoch(
            epoch=0, gps_tow=518400.0, doppler_hz=_spoofing_doppler(rng)
        )
        assert result.decoder.decision != DecisionClass.NOMINAL

    def test_spoofing_decision_is_spoofing(self) -> None:
        rng = np.random.default_rng(0)
        pipeline = _make_pipeline("spftest2")
        result = pipeline.process_epoch(
            epoch=0, gps_tow=518400.0, doppler_hz=_spoofing_doppler(rng, common_bias=5.0)
        )
        assert result.decoder.decision == DecisionClass.SPOOFING

    def test_spoofing_syndrome_edge_present(self) -> None:
        rng = np.random.default_rng(0)
        pipeline = _make_pipeline("spftest3")
        result = pipeline.process_epoch(
            epoch=0, gps_tow=518400.0, doppler_hz=_spoofing_doppler(rng)
        )
        assert result.n_syndrome_edges > 0
        types = [e.constraint_type for e in pipeline.syndrome_graph.edges_at_epoch(0)]
        assert ConstraintType.CROSS_SAT_DOPPLER in types

    def test_spoofing_certificate_has_failed_checks(self) -> None:
        rng = np.random.default_rng(0)
        pipeline = _make_pipeline("spftest4")
        result = pipeline.process_epoch(
            epoch=0, gps_tow=518400.0, doppler_hz=_spoofing_doppler(rng)
        )
        cert = result.certificate
        assert cert.is_valid()
        assert len(cert.checks_failed) > 0
        assert "cross_sat_doppler" in cert.checks_failed

    def test_spoofing_anomaly_beliefs_high(self) -> None:
        rng = np.random.default_rng(0)
        pipeline = _make_pipeline("spftest5")
        result = pipeline.process_epoch(
            epoch=0, gps_tow=518400.0, doppler_hz=_spoofing_doppler(rng)
        )
        # With coherent bias, all sats should have high anomaly beliefs.
        assert result.decoder.n_anomalous == N_SATS
        for belief in result.decoder.anomaly_beliefs:
            assert belief > 0.5


# ---------------------------------------------------------------------------
# Hardware fault scenario
# ---------------------------------------------------------------------------


class TestHardwareFaultScenario:
    def test_raim_failure_without_coherent_shift_gives_hw_fault(self) -> None:
        """GEOMETRY_RAIM active + CROSS_SAT_DOPPLER inactive → HARDWARE_FAULT."""
        rng = np.random.default_rng(10)
        pipeline = _make_pipeline("hwtest1")
        # Low coherent SNR (single outlier, not common mode).
        dop = _hardware_fault_doppler(rng, fault_sat_idx=0, bias=20.0)
        result = pipeline.process_epoch(
            epoch=0,
            gps_tow=518400.0,
            doppler_hz=dop,
            raim_chi2=50.0,  # High RAIM chi² (large outlier)
        )
        assert result.decoder.decision == DecisionClass.HARDWARE_FAULT

    def test_hw_fault_certificate_valid(self) -> None:
        rng = np.random.default_rng(10)
        pipeline = _make_pipeline("hwtest2")
        dop = _hardware_fault_doppler(rng)
        result = pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=dop, raim_chi2=50.0)
        assert result.certificate.is_valid()
        assert "geometry_raim" in result.certificate.checks_failed


# ---------------------------------------------------------------------------
# Multi-epoch sequence
# ---------------------------------------------------------------------------


class TestMultiEpochSequence:
    def test_sequential_epochs_produce_correct_epoch_ids(self) -> None:
        rng = np.random.default_rng(1)
        pipeline = _make_pipeline("seqtest1")
        for t in range(5):
            result = pipeline.process_epoch(
                epoch=t, gps_tow=518400.0 + t * 30.0, doppler_hz=_nominal_doppler(rng)
            )
            assert result.epoch == t
            assert result.certificate.epoch == t

    def test_syndrome_graph_grows_with_spoofing(self) -> None:
        rng = np.random.default_rng(2)
        pipeline = _make_pipeline("seqtest2")
        prev_count = 0
        for t in range(4):
            pipeline.process_epoch(
                epoch=t,
                gps_tow=518400.0 + t * 30.0,
                doppler_hz=_spoofing_doppler(rng),
            )
            curr_count = pipeline.syndrome_graph.total_edges()
            assert curr_count > prev_count  # spoofing always adds at least one edge
            prev_count = curr_count

    def test_duplicate_epoch_raises(self) -> None:
        rng = np.random.default_rng(3)
        pipeline = _make_pipeline("seqtest3")
        pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=_nominal_doppler(rng))
        with pytest.raises(ValueError, match="already processed"):
            pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=_nominal_doppler(rng))

    def test_run_id_consistent_across_epochs(self) -> None:
        rng = np.random.default_rng(4)
        pipeline = _make_pipeline("consistid")
        for t in range(3):
            result = pipeline.process_epoch(
                epoch=t, gps_tow=518400.0 + t * 30.0, doppler_hz=_nominal_doppler(rng)
            )
            assert result.certificate.run_id == "consistid"


# ---------------------------------------------------------------------------
# Signal layer output
# ---------------------------------------------------------------------------


class TestSignalLayerOutput:
    def test_signal_features_count_matches_n_sats(self) -> None:
        rng = np.random.default_rng(5)
        pipeline = _make_pipeline("sigtest1")
        result = pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=_nominal_doppler(rng))
        assert len(result.signal_features) == N_SATS

    def test_signal_feature_doppler_preserved(self) -> None:
        rng = np.random.default_rng(5)
        pipeline = _make_pipeline("sigtest2")
        dop = _nominal_doppler(rng)
        result = pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=dop)
        for i, feat in enumerate(result.signal_features):
            assert feat.feature_vector[0] == pytest.approx(float(dop[i]))

    def test_signal_feature_satellite_ids_match(self) -> None:
        rng = np.random.default_rng(5)
        pipeline = _make_pipeline("sigtest3")
        result = pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=_nominal_doppler(rng))
        for feat, expected_id in zip(result.signal_features, SAT_IDS):
            assert feat.satellite_id == expected_id


# ---------------------------------------------------------------------------
# Certificate invariant
# ---------------------------------------------------------------------------


class TestCertificateInvariant:
    def test_nominal_certificate_is_valid(self) -> None:
        rng = np.random.default_rng(42)
        pipeline = _make_pipeline("certtest1")
        result = pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=_nominal_doppler(rng))
        assert result.certificate.is_valid()

    def test_spoofing_certificate_is_valid(self) -> None:
        rng = np.random.default_rng(0)
        pipeline = _make_pipeline("certtest2")
        result = pipeline.process_epoch(
            epoch=0, gps_tow=518400.0, doppler_hz=_spoofing_doppler(rng)
        )
        assert result.certificate.is_valid()

    def test_certificate_evidence_digest_deterministic(self) -> None:
        """Same inputs → same evidence digest."""
        dop = np.ones(N_SATS) * 5.0

        p1 = LayeredPipeline(SAT_IDS, run_id="fixed_id")
        p2 = LayeredPipeline(SAT_IDS, run_id="fixed_id")
        r1 = p1.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=dop)
        r2 = p2.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=dop)
        assert r1.certificate.evidence_digest == r2.certificate.evidence_digest

    def test_build_certificate_non_nominal_needs_checks_failed(self) -> None:
        """Structural invariant: non-NOMINAL decision requires non-empty checks_failed."""
        # Manually construct a spoofing decoder result with a syndrome edge.
        decoder_result = DecoderResult(
            satellite_ids=tuple(SAT_IDS),
            anomaly_beliefs=tuple([0.95] * N_SATS),
            decision=DecisionClass.SPOOFING,
            decision_score=0.95,
            n_anomalous=N_SATS,
            bp_converged=True,
            active_constraints=frozenset([ConstraintType.CROSS_SAT_DOPPLER]),
        )
        node = SyndromeNode("ALL", 0)
        edge = SyndromeEdge(
            node_a=node,
            node_b=node,
            constraint_type=ConstraintType.CROSS_SAT_DOPPLER,
            value=60000.0,
            threshold=5.0,
            epoch=0,
            digest="placeholder",
        )
        cert = build_certificate(
            epoch=0,
            gps_tow=518400.0,
            run_id="invtest",
            decoder_result=decoder_result,
            syndrome_edges=[edge],
        )
        assert cert.is_valid()
        assert len(cert.checks_failed) > 0
        assert "cross_sat_doppler" in cert.checks_failed

    def test_certificate_invariants_plus_failed_eq_all_invariants(self) -> None:
        """invariants_satisfied ∪ checks_failed ⊆ _ALL_INVARIANTS."""
        from gnss.certificate import _ALL_INVARIANTS

        rng = np.random.default_rng(0)
        pipeline = _make_pipeline("parttest")
        result = pipeline.process_epoch(
            epoch=0, gps_tow=518400.0, doppler_hz=_spoofing_doppler(rng)
        )
        cert = result.certificate
        all_inv = set(_ALL_INVARIANTS)
        for name in cert.invariants_satisfied:
            assert name in all_inv
        for name in cert.checks_failed:
            assert name in all_inv


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_wrong_n_sats_raises(self) -> None:
        pipeline = _make_pipeline("valerr1")
        with pytest.raises(ValueError, match="n_satellites"):
            pipeline.process_epoch(epoch=0, gps_tow=518400.0, doppler_hz=np.zeros(3))

    def test_run_id_property(self) -> None:
        pipeline = LayeredPipeline(SAT_IDS, run_id="myrunid1")
        assert pipeline.run_id == "myrunid1"

    def test_n_satellites_property(self) -> None:
        pipeline = _make_pipeline()
        assert pipeline.n_satellites == N_SATS
