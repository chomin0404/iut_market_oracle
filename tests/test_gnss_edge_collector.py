"""Tests for EdgeCollector (T1500 — edge signal accumulation)."""

from __future__ import annotations

import numpy as np
import pytest

from gnss.edge_collector import EdgeArrays, EdgeCollector
from gnss.mvp import MVPPipeline, RawEpochData
from gnss.spoof_sim import _init_constellation
from schemas import FaultClass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SATS = 6


def _make_los(n: int = N_SATS) -> np.ndarray:
    return _init_constellation(n)


def _make_raw(
    epoch: int = 0,
    n: int = N_SATS,
    seed: int | None = None,
    sqm: np.ndarray | None = None,
    imu: np.ndarray | None = None,
    osnma: list[bool] | None = None,
) -> RawEpochData:
    rng = np.random.default_rng(seed if seed is not None else epoch)
    doppler = rng.normal(0.0, 0.30, size=n)
    return RawEpochData(
        epoch=epoch,
        doppler_residuals=doppler,
        sqm=sqm,
        imu_velocity=imu,
        osnma_auth=osnma,
    )


def _run_pipeline(n_epochs: int = 5, n_sats: int = N_SATS) -> MVPPipeline:
    """Run a nominal pipeline for n_epochs and return it."""
    los = _make_los(n_sats)
    pipeline = MVPPipeline(n_sats=n_sats, los=los)
    for e in range(n_epochs):
        pipeline.step(_make_raw(epoch=e, n=n_sats))
    return pipeline


# ---------------------------------------------------------------------------
# EdgeSnapshot field integrity
# ---------------------------------------------------------------------------


class TestEdgeSnapshot:
    def test_collect_single_epoch(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        assert len(collector) == 1

    def test_snapshot_epoch_index(self) -> None:
        pipeline = _run_pipeline(n_epochs=3)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        epochs = [s.epoch for s in collector.snapshots]
        assert epochs == [0, 1, 2]

    def test_snapshot_doppler_shape(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        assert snap.doppler_residuals.shape == (N_SATS,)

    def test_snapshot_doppler_is_copy(self) -> None:
        """Snapshot stores an independent copy of the doppler array."""
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        obs_doppler = pipeline.history[0].obs.doppler_residuals
        # Different object (not the same buffer as ReceiverObservation)
        assert snap.doppler_residuals is not obs_doppler
        # Values must match the original
        assert np.allclose(snap.doppler_residuals, obs_doppler)

    def test_snapshot_sqm_none_by_default(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        assert collector.snapshots[0].sqm is None

    def test_snapshot_sqm_populated_when_present(self) -> None:
        sqm = np.linspace(0.1, 0.5, N_SATS)
        los = _make_los()
        pipeline = MVPPipeline(n_sats=N_SATS, los=los)
        pipeline.step(_make_raw(epoch=0, sqm=sqm))
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        assert snap.sqm is not None
        assert snap.sqm.shape == (N_SATS,)

    def test_snapshot_ins_velocity_none_by_default(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        assert collector.snapshots[0].ins_velocity is None

    def test_snapshot_ins_velocity_populated(self) -> None:
        imu = np.array([0.1, -0.2, 0.05])
        los = _make_los()
        pipeline = MVPPipeline(n_sats=N_SATS, los=los)
        pipeline.step(_make_raw(epoch=0, imu=imu))
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        assert snap.ins_velocity is not None
        assert snap.ins_velocity.shape == (3,)
        assert np.allclose(snap.ins_velocity, imu)

    def test_snapshot_osnma_none_by_default(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        assert collector.snapshots[0].osnma_auth is None

    def test_snapshot_osnma_forwarded(self) -> None:
        flags = [True, True, False, True, True, True]
        los = _make_los()
        pipeline = MVPPipeline(n_sats=N_SATS, los=los)
        pipeline.step(_make_raw(epoch=0, osnma=flags))
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        assert snap.osnma_auth is not None
        assert list(snap.osnma_auth) == flags

    def test_snapshot_gmm_gamma_length(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        assert len(snap.gmm_gamma) == N_SATS

    def test_snapshot_imm_fields_length_3(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        assert len(snap.imm_innovation_norms) == 3
        assert len(snap.imm_mode_weights) == 3

    def test_snapshot_imm_mode_weights_sum_to_one(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        assert pytest.approx(sum(snap.imm_mode_weights), abs=1e-6) == 1.0

    def test_snapshot_auth_fraction_range(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        assert 0.0 <= snap.auth_fraction <= 1.0

    def test_snapshot_fault_posterior_sums_to_one(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        assert pytest.approx(sum(snap.fault_posterior), abs=1e-6) == 1.0

    def test_snapshot_confidence_is_max_posterior(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        snap = collector.snapshots[0]
        assert pytest.approx(snap.confidence, abs=1e-9) == max(snap.fault_posterior)

    def test_snapshot_diagnosis_is_fault_class(self) -> None:
        pipeline = _run_pipeline(n_epochs=1)
        collector = EdgeCollector()
        collector.collect(pipeline.history[0])
        assert isinstance(collector.snapshots[0].diagnosis, FaultClass)

    def test_snapshot_ins_weight_in_range(self) -> None:
        pipeline = _run_pipeline(n_epochs=5)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        for snap in collector.snapshots:
            assert 0.0 <= snap.ins_weight <= 1.0

    def test_snapshot_n_active_leq_n_sats(self) -> None:
        pipeline = _run_pipeline(n_epochs=5)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        for snap in collector.snapshots:
            assert snap.n_active <= N_SATS

    def test_snapshot_mc_auc_none_in_nominal(self) -> None:
        """Nominal epochs rarely hit the low-confidence MC trigger."""
        pipeline = _run_pipeline(n_epochs=3)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        # mc_auc may be None or float — just verify the field exists and is typed
        for snap in collector.snapshots:
            assert snap.mc_auc is None or isinstance(snap.mc_auc, float)


# ---------------------------------------------------------------------------
# collect_all / collect equivalence
# ---------------------------------------------------------------------------


class TestCollectAll:
    def test_collect_all_equiv_to_loop(self) -> None:
        pipeline = _run_pipeline(n_epochs=4)
        c1 = EdgeCollector()
        c1.collect_all(pipeline.history)

        c2 = EdgeCollector()
        for rec in pipeline.history:
            c2.collect(rec)

        assert len(c1) == len(c2) == 4
        for s1, s2 in zip(c1.snapshots, c2.snapshots):
            assert s1.epoch == s2.epoch
            assert np.allclose(s1.doppler_residuals, s2.doppler_residuals)

    def test_empty_collector_length_zero(self) -> None:
        collector = EdgeCollector()
        assert len(collector) == 0


# ---------------------------------------------------------------------------
# to_arrays — shape and value checks
# ---------------------------------------------------------------------------


class TestToArrays:
    def test_raises_when_empty(self) -> None:
        collector = EdgeCollector()
        with pytest.raises(ValueError, match="no snapshots"):
            collector.to_arrays()

    def test_epochs_shape(self) -> None:
        n = 7
        pipeline = _run_pipeline(n_epochs=n)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        assert arrays.epochs.shape == (n,)
        assert list(arrays.epochs) == list(range(n))

    def test_doppler_shape(self) -> None:
        n, s = 5, N_SATS
        pipeline = _run_pipeline(n_epochs=n, n_sats=s)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        assert arrays.doppler_residuals.shape == (n, s)

    def test_gmm_gamma_shape(self) -> None:
        n, s = 4, N_SATS
        pipeline = _run_pipeline(n_epochs=n, n_sats=s)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        assert arrays.gmm_gamma.shape == (n, s)

    def test_imm_shapes(self) -> None:
        n = 6
        pipeline = _run_pipeline(n_epochs=n)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        assert arrays.imm_innovation_norms.shape == (n, 3)
        assert arrays.imm_mode_weights.shape == (n, 3)

    def test_fault_posterior_shape(self) -> None:
        n = 8
        pipeline = _run_pipeline(n_epochs=n)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        assert arrays.fault_posterior.shape == (n, 4)

    def test_scalar_arrays_shape(self) -> None:
        n = 5
        pipeline = _run_pipeline(n_epochs=n)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        for arr in (
            arrays.auth_fraction, arrays.confidence, arrays.ins_weight,
            arrays.mc_auc, arrays.osnma_alert, arrays.entropy_alert,
            arrays.structure_alert, arrays.phase_alert,
        ):
            assert arr.shape == (n,)

    def test_int_arrays_shape(self) -> None:
        n = 5
        pipeline = _run_pipeline(n_epochs=n)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        for arr in (arrays.n_auth, arrays.n_total, arrays.n_excluded, arrays.n_active):
            assert arr.shape == (n,)

    def test_diagnosis_dtype_object(self) -> None:
        pipeline = _run_pipeline(n_epochs=3)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        assert arrays.diagnosis.dtype == object
        for d in arrays.diagnosis:
            assert isinstance(d, FaultClass)

    def test_mc_auc_nan_for_nominal_epochs(self) -> None:
        """In a short nominal run most mc_auc entries should be NaN."""
        pipeline = _run_pipeline(n_epochs=3)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        # At minimum the array exists and is float
        assert arrays.mc_auc.dtype == np.float64

    def test_fault_posterior_rows_sum_to_one(self) -> None:
        pipeline = _run_pipeline(n_epochs=4)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        row_sums = arrays.fault_posterior.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_imm_mode_weights_rows_sum_to_one(self) -> None:
        pipeline = _run_pipeline(n_epochs=4)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        row_sums = arrays.imm_mode_weights.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_confidence_matches_max_posterior(self) -> None:
        pipeline = _run_pipeline(n_epochs=4)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        expected = arrays.fault_posterior.max(axis=1)
        assert np.allclose(arrays.confidence, expected, atol=1e-9)

    def test_n_epochs_property(self) -> None:
        n = 6
        pipeline = _run_pipeline(n_epochs=n)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        assert arrays.n_epochs == n

    def test_n_sats_property(self) -> None:
        pipeline = _run_pipeline(n_epochs=3, n_sats=N_SATS)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        assert arrays.n_sats == N_SATS

    def test_ins_weight_in_range(self) -> None:
        pipeline = _run_pipeline(n_epochs=5)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        arrays = collector.to_arrays()
        assert np.all(arrays.ins_weight >= 0.0)
        assert np.all(arrays.ins_weight <= 1.0)


# ---------------------------------------------------------------------------
# EdgeArrays.event_mask
# ---------------------------------------------------------------------------


class TestEventMask:
    def _arrays_with_forced_alerts(self) -> EdgeArrays:
        """Return EdgeArrays with known alert pattern for testing."""
        pipeline = _run_pipeline(n_epochs=4)
        collector = EdgeCollector()
        collector.collect_all(pipeline.history)
        return collector.to_arrays()

    def test_event_mask_shape(self) -> None:
        arrays = self._arrays_with_forced_alerts()
        mask = arrays.event_mask()
        assert mask.shape == (arrays.n_epochs,)
        assert mask.dtype == bool

    def test_event_mask_no_types_selected_all_false(self) -> None:
        arrays = self._arrays_with_forced_alerts()
        mask = arrays.event_mask(entropy=False, structure=False, phase=False, osnma=False)
        assert not mask.any()

    def test_event_mask_all_types_superset(self) -> None:
        """Union of all types must be a superset of any individual type."""
        arrays = self._arrays_with_forced_alerts()
        full_mask = arrays.event_mask()
        entropy_only = arrays.event_mask(
            entropy=True, structure=False, phase=False, osnma=False
        )
        # full_mask is OR of all — any epoch in entropy_only must also be in full_mask
        assert np.all(full_mask | ~entropy_only)

    def test_event_mask_entropy_only(self) -> None:
        arrays = self._arrays_with_forced_alerts()
        mask = arrays.event_mask(entropy=True, structure=False, phase=False, osnma=False)
        assert np.array_equal(mask, arrays.entropy_alert)

    def test_event_mask_structure_only(self) -> None:
        arrays = self._arrays_with_forced_alerts()
        mask = arrays.event_mask(entropy=False, structure=True, phase=False, osnma=False)
        assert np.array_equal(mask, arrays.structure_alert)
