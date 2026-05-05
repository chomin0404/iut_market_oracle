"""Tests for gnss.action_engine — SatelliteScorer, FailsafeManager, AlertBuilder."""

from __future__ import annotations

import numpy as np
import pytest

from gnss.action_engine import (
    _FAILSAFE_RECOVERY_EPOCHS,
    _SPOOFING_DEGRADED_THRESH,
    _SPOOFING_INS_ONLY_THRESH,
    DOWNWEIGHT_THRESH,
    HARD_EXCLUDE_THRESH,
    AlertBuilder,
    AlertEvent,
    AlertLevel,
    FailsafeLevel,
    FailsafeManager,
    FailsafeState,
    SatelliteScorer,
)

# ---------------------------------------------------------------------------
# SatelliteScorer
# ---------------------------------------------------------------------------


class TestSatelliteScorerWeightValidation:
    def test_default_weights_sum_to_one(self) -> None:
        scorer = SatelliteScorer()
        assert scorer is not None  # construction succeeds

    def test_invalid_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            SatelliteScorer(w_gmm=0.5, w_sqm=0.5, w_osnma=0.5)

    def test_custom_valid_weights(self) -> None:
        SatelliteScorer(w_gmm=0.70, w_sqm=0.20, w_osnma=0.10)  # no exception


class TestSatelliteScorerGmmOnly:
    """Score with only GMM gamma provided."""

    def test_zero_gamma_zero_score(self) -> None:
        scorer = SatelliteScorer()
        scores = scorer.score(gmm_gamma=(0.0, 0.0, 0.0), n_sats=3)
        assert np.allclose(scores, 0.0)

    def test_unit_gamma_gives_gmm_weight(self) -> None:
        scorer = SatelliteScorer(w_gmm=0.60, w_sqm=0.25, w_osnma=0.15)
        scores = scorer.score(gmm_gamma=(1.0, 1.0), n_sats=2)
        # Only GMM term (SQM and OSNMA absent) → score = 0.60 * 1.0 = 0.60
        assert np.allclose(scores, 0.60)

    def test_score_clipped_to_one(self) -> None:
        scorer = SatelliteScorer()
        # gamma > 1 should be clipped before weighting
        scores = scorer.score(gmm_gamma=(2.0,), n_sats=1)
        assert scores[0] <= 1.0

    def test_score_clipped_to_zero(self) -> None:
        scorer = SatelliteScorer()
        scores = scorer.score(gmm_gamma=(-1.0,), n_sats=1)
        assert scores[0] >= 0.0

    def test_shape_matches_n_sats(self) -> None:
        scorer = SatelliteScorer()
        scores = scorer.score(gmm_gamma=(0.1, 0.2, 0.3, 0.4), n_sats=4)
        assert scores.shape == (4,)
        assert scores.dtype == np.float64


class TestSatelliteScorerSqm:
    def test_sqm_above_thresh_adds_w_sqm(self) -> None:
        scorer = SatelliteScorer(w_gmm=0.60, w_sqm=0.25, w_osnma=0.15, sqm_thresh=0.70)
        # gamma=0, sqm=0.9 > 0.70 → score = 0.25
        scores = scorer.score(gmm_gamma=(0.0,), n_sats=1, sqm=np.array([0.9]))
        assert pytest.approx(scores[0], abs=1e-9) == 0.25

    def test_sqm_below_thresh_no_contribution(self) -> None:
        scorer = SatelliteScorer(w_gmm=0.60, w_sqm=0.25, w_osnma=0.15, sqm_thresh=0.70)
        scores = scorer.score(gmm_gamma=(0.0,), n_sats=1, sqm=np.array([0.5]))
        assert pytest.approx(scores[0], abs=1e-9) == 0.0

    def test_sqm_exactly_at_thresh_no_contribution(self) -> None:
        scorer = SatelliteScorer(w_gmm=0.60, w_sqm=0.25, w_osnma=0.15, sqm_thresh=0.70)
        # strict > threshold
        scores = scorer.score(gmm_gamma=(0.0,), n_sats=1, sqm=np.array([0.70]))
        assert pytest.approx(scores[0], abs=1e-9) == 0.0


class TestSatelliteScorerOsnma:
    def test_failed_auth_adds_w_osnma(self) -> None:
        scorer = SatelliteScorer(w_gmm=0.60, w_sqm=0.25, w_osnma=0.15)
        # gamma=0, no SQM, auth=False → score = 0.15*(1-0) = 0.15
        scores = scorer.score(gmm_gamma=(0.0,), n_sats=1, osnma_auth=[False])
        assert pytest.approx(scores[0], abs=1e-9) == 0.15

    def test_passed_auth_no_osnma_contribution(self) -> None:
        scorer = SatelliteScorer(w_gmm=0.60, w_sqm=0.25, w_osnma=0.15)
        scores = scorer.score(gmm_gamma=(0.0,), n_sats=1, osnma_auth=[True])
        assert pytest.approx(scores[0], abs=1e-9) == 0.0

    def test_all_failed_auth_max_osnma_term(self) -> None:
        scorer = SatelliteScorer(w_gmm=0.0, w_sqm=0.0, w_osnma=1.0)
        scores = scorer.score(
            gmm_gamma=(0.0, 0.0, 0.0),
            n_sats=3,
            osnma_auth=[False, False, False],
        )
        assert np.allclose(scores, 1.0)


class TestSatelliteScorerFull:
    def test_all_sources_sum(self) -> None:
        scorer = SatelliteScorer(w_gmm=0.60, w_sqm=0.25, w_osnma=0.15, sqm_thresh=0.70)
        # gamma=1.0, sqm=0.9>0.7, auth=False → 0.60+0.25+0.15 = 1.0
        scores = scorer.score(
            gmm_gamma=(1.0,),
            n_sats=1,
            sqm=np.array([0.9]),
            osnma_auth=[False],
        )
        assert pytest.approx(scores[0], abs=1e-9) == 1.0

    def test_score_between_zero_and_one(self) -> None:
        scorer = SatelliteScorer()
        rng = np.random.default_rng(7)
        gamma = tuple(rng.uniform(0, 1, 8).tolist())
        sqm = rng.uniform(0, 1, 8)
        osnma = list(rng.integers(0, 2, 8).astype(bool))
        scores = scorer.score(gmm_gamma=gamma, n_sats=8, sqm=sqm, osnma_auth=osnma)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


# ---------------------------------------------------------------------------
# FailsafeManager — target level logic
# ---------------------------------------------------------------------------


class TestFailsafeManagerInit:
    def test_initial_level_is_nominal(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        assert mgr.current_level == FailsafeLevel.NOMINAL

    def test_reset_returns_to_nominal(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        mgr.update(n_active=0, spoofing_prob=0.0, entropy_alert=False, osnma_all_failed=False)
        mgr.reset()
        assert mgr.current_level == FailsafeLevel.NOMINAL


class TestFailsafeManagerDescent:
    """Descent (worsening) must be immediate."""

    def _nom(self, mgr: FailsafeManager, **kw: object) -> FailsafeState:
        return mgr.update(spoofing_prob=0.0, entropy_alert=False, osnma_all_failed=False, **kw)  # type: ignore[arg-type]

    def test_zero_active_triggers_dead_reckoning(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = self._nom(mgr, n_active=0)
        assert state.level == FailsafeLevel.DEAD_RECKONING

    def test_osnma_all_failed_triggers_dead_reckoning(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = mgr.update(
            n_active=5, spoofing_prob=0.0, entropy_alert=False, osnma_all_failed=True
        )
        assert state.level == FailsafeLevel.DEAD_RECKONING

    def test_below_min_sats_triggers_ins_only(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        # n_active=3 < min_sats=4 → INS_ONLY
        state = self._nom(mgr, n_active=3)
        assert state.level == FailsafeLevel.INS_ONLY

    def test_high_spoof_prob_triggers_ins_only(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = mgr.update(
            n_active=6,
            spoofing_prob=_SPOOFING_INS_ONLY_THRESH + 0.01,
            entropy_alert=False,
            osnma_all_failed=False,
        )
        assert state.level == FailsafeLevel.INS_ONLY

    def test_moderate_spoof_prob_triggers_degraded(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = mgr.update(
            n_active=6,
            spoofing_prob=_SPOOFING_DEGRADED_THRESH + 0.01,
            entropy_alert=False,
            osnma_all_failed=False,
        )
        assert state.level == FailsafeLevel.DEGRADED

    def test_entropy_alert_triggers_degraded(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = mgr.update(
            n_active=6, spoofing_prob=0.0, entropy_alert=True, osnma_all_failed=False
        )
        assert state.level == FailsafeLevel.DEGRADED

    def test_n_active_at_min_plus_one_triggers_degraded(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        # n_active = min_sats (4) < min_sats+1 (5) → DEGRADED
        state = self._nom(mgr, n_active=4)
        assert state.level == FailsafeLevel.DEGRADED

    def test_sufficient_sats_clean_signal_nominal(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = self._nom(mgr, n_active=6)
        assert state.level == FailsafeLevel.NOMINAL

    def test_descent_is_immediate(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        for _ in range(3):
            self._nom(mgr, n_active=6)
        # One bad epoch → immediate descent
        state = self._nom(mgr, n_active=0)
        assert state.level == FailsafeLevel.DEAD_RECKONING
        assert state.transitioned is True


class TestFailsafeManagerRecovery:
    """Recovery (ascent) requires _FAILSAFE_RECOVERY_EPOCHS consecutive eligible epochs."""

    def _descend_to_ins_only(self, mgr: FailsafeManager) -> None:
        mgr.update(
            n_active=6,
            spoofing_prob=_SPOOFING_INS_ONLY_THRESH + 0.01,
            entropy_alert=False,
            osnma_all_failed=False,
        )
        assert mgr.current_level == FailsafeLevel.INS_ONLY

    def test_single_recovery_epoch_not_enough(self) -> None:
        mgr = FailsafeManager(min_sats=4, recovery_thresh=3)
        self._descend_to_ins_only(mgr)
        state = mgr.update(
            n_active=6, spoofing_prob=0.0, entropy_alert=False, osnma_all_failed=False
        )
        assert state.level == FailsafeLevel.INS_ONLY  # not yet recovered

    def test_recovery_after_threshold_epochs(self) -> None:
        mgr = FailsafeManager(min_sats=4, recovery_thresh=3)
        self._descend_to_ins_only(mgr)
        for _ in range(3):
            state = mgr.update(
                n_active=6, spoofing_prob=0.0, entropy_alert=False, osnma_all_failed=False
            )
        # After 3 consecutive recovery epochs, should have ascended
        assert state.level != FailsafeLevel.INS_ONLY

    def test_recovery_streak_resets_on_bad_epoch(self) -> None:
        mgr = FailsafeManager(min_sats=4, recovery_thresh=3)
        self._descend_to_ins_only(mgr)
        # 2 recovery epochs, then 1 bad epoch
        for _ in range(2):
            mgr.update(
                n_active=6, spoofing_prob=0.0, entropy_alert=False, osnma_all_failed=False
            )
        mgr.update(
            n_active=6,
            spoofing_prob=_SPOOFING_INS_ONLY_THRESH + 0.01,
            entropy_alert=False,
            osnma_all_failed=False,
        )
        # Bad epoch resets streak; still at INS_ONLY (or descended further and back)
        assert mgr.current_level != FailsafeLevel.NOMINAL

    def test_recovery_streak_field_increments(self) -> None:
        mgr = FailsafeManager(min_sats=4, recovery_thresh=5)
        self._descend_to_ins_only(mgr)
        s1 = mgr.update(
            n_active=6, spoofing_prob=0.0, entropy_alert=False, osnma_all_failed=False
        )
        s2 = mgr.update(
            n_active=6, spoofing_prob=0.0, entropy_alert=False, osnma_all_failed=False
        )
        assert s2.recovery_streak == s1.recovery_streak + 1


class TestFailsafeManagerInsWeightBounds:
    def _upd(self, mgr: FailsafeManager, n: int, **kw: object) -> FailsafeState:
        return mgr.update(  # type: ignore[arg-type]
            n_active=n, spoofing_prob=0.0, entropy_alert=False, osnma_all_failed=False, **kw
        )

    def test_nominal_floor_zero(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = self._upd(mgr, 6)
        assert state.ins_weight_floor == 0.0
        assert state.ins_weight_ceil == 1.0

    def test_degraded_bounds(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = mgr.update(
            n_active=6, spoofing_prob=0.0, entropy_alert=True, osnma_all_failed=False
        )
        assert state.level == FailsafeLevel.DEGRADED
        assert state.ins_weight_floor == 0.45
        assert state.ins_weight_ceil == 0.70

    def test_ins_only_fixed_weight(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = mgr.update(
            n_active=6,
            spoofing_prob=_SPOOFING_INS_ONLY_THRESH + 0.01,
            entropy_alert=False,
            osnma_all_failed=False,
        )
        assert state.level == FailsafeLevel.INS_ONLY
        assert state.ins_weight_floor == 0.90
        assert state.ins_weight_ceil == 0.90

    def test_dead_reckoning_fixed_weight(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = self._upd(mgr, 0)
        assert state.level == FailsafeLevel.DEAD_RECKONING
        assert state.ins_weight_floor == 1.0
        assert state.ins_weight_ceil == 1.0


class TestFailsafeManagerStateFields:
    def _upd(self, mgr: FailsafeManager, n: int) -> FailsafeState:
        return mgr.update(
            n_active=n, spoofing_prob=0.0, entropy_alert=False, osnma_all_failed=False
        )

    def test_transitioned_true_on_level_change(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        state = self._upd(mgr, 0)
        assert state.transitioned is True

    def test_transitioned_false_when_same_level(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        self._upd(mgr, 6)
        state = self._upd(mgr, 6)
        assert state.transitioned is False

    def test_epochs_in_level_increments(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        s1 = self._upd(mgr, 6)
        s2 = self._upd(mgr, 6)
        assert s2.epochs_in_level == s1.epochs_in_level + 1

    def test_epochs_in_level_resets_on_transition(self) -> None:
        mgr = FailsafeManager(min_sats=4)
        self._upd(mgr, 6)
        self._upd(mgr, 6)
        state = self._upd(mgr, 0)
        assert state.epochs_in_level == 1


# ---------------------------------------------------------------------------
# AlertBuilder
# ---------------------------------------------------------------------------


def _make_failsafe(
    level: FailsafeLevel = FailsafeLevel.NOMINAL,
) -> FailsafeState:
    from gnss.action_engine import _FAILSAFE_INS_BOUNDS

    floor, ceil_ = _FAILSAFE_INS_BOUNDS[level.value]
    return FailsafeState(
        level=level,
        epochs_in_level=1,
        recovery_streak=0,
        transitioned=False,
        ins_weight_floor=floor,
        ins_weight_ceil=ceil_,
    )


class TestAlertBuilderLevel:
    def _build(
        self,
        spoof_prob: float = 0.0,
        n_sources: int = 0,
        failsafe_level: FailsafeLevel = FailsafeLevel.NOMINAL,
        mc_auc: float | None = None,
    ) -> AlertEvent:
        builder = AlertBuilder()
        fp = (1.0 - spoof_prob, 0.0, 0.0, spoof_prob)
        entropy = n_sources >= 1
        osnma = n_sources >= 2
        phase = n_sources >= 3
        structure = n_sources >= 4
        return builder.build(
            epoch=0,
            fault_posterior=fp,
            entropy_alert=entropy,
            osnma_alert=osnma,
            phase_alert=phase,
            structure_alert=structure,
            failsafe=_make_failsafe(failsafe_level),
            n_active=5,
            mc_auc=mc_auc,
        )

    def test_no_alerts_is_info(self) -> None:
        evt = self._build(spoof_prob=0.0, n_sources=0)
        assert evt.level == AlertLevel.INFO

    def test_one_source_is_caution(self) -> None:
        evt = self._build(spoof_prob=0.0, n_sources=1)
        assert evt.level == AlertLevel.CAUTION

    def test_two_sources_is_warning(self) -> None:
        evt = self._build(spoof_prob=0.0, n_sources=2)
        assert evt.level == AlertLevel.WARNING

    def test_spoof_above_degraded_thresh_is_warning(self) -> None:
        evt = self._build(spoof_prob=_SPOOFING_DEGRADED_THRESH + 0.01, n_sources=0)
        assert evt.level == AlertLevel.WARNING

    def test_spoof_above_ins_only_thresh_is_critical(self) -> None:
        evt = self._build(spoof_prob=_SPOOFING_INS_ONLY_THRESH + 0.01, n_sources=0)
        assert evt.level == AlertLevel.CRITICAL

    def test_ins_only_failsafe_is_critical(self) -> None:
        evt = self._build(spoof_prob=0.0, failsafe_level=FailsafeLevel.INS_ONLY)
        assert evt.level == AlertLevel.CRITICAL

    def test_dead_reckoning_failsafe_is_critical(self) -> None:
        evt = self._build(spoof_prob=0.0, failsafe_level=FailsafeLevel.DEAD_RECKONING)
        assert evt.level == AlertLevel.CRITICAL


class TestAlertBuilderFields:
    def _builder(self) -> AlertBuilder:
        return AlertBuilder()

    def test_epoch_forwarded(self) -> None:
        builder = AlertBuilder()
        fp = (1.0, 0.0, 0.0, 0.0)
        evt = builder.build(
            epoch=42,
            fault_posterior=fp,
            entropy_alert=False,
            osnma_alert=False,
            phase_alert=False,
            structure_alert=False,
            failsafe=_make_failsafe(),
            n_active=5,
            mc_auc=None,
        )
        assert evt.epoch == 42

    def test_sources_tuple_correct(self) -> None:
        builder = AlertBuilder()
        fp = (0.5, 0.0, 0.0, 0.5)
        evt = builder.build(
            epoch=0,
            fault_posterior=fp,
            entropy_alert=True,
            osnma_alert=False,
            phase_alert=True,
            structure_alert=False,
            failsafe=_make_failsafe(),
            n_active=5,
            mc_auc=None,
        )
        assert "entropy" in evt.sources
        assert "phase" in evt.sources
        assert "osnma" not in evt.sources
        assert "structure" not in evt.sources

    def test_n_active_forwarded(self) -> None:
        builder = AlertBuilder()
        fp = (1.0, 0.0, 0.0, 0.0)
        evt = builder.build(
            epoch=0,
            fault_posterior=fp,
            entropy_alert=False,
            osnma_alert=False,
            phase_alert=False,
            structure_alert=False,
            failsafe=_make_failsafe(),
            n_active=3,
            mc_auc=None,
        )
        assert evt.n_active == 3

    def test_mc_auc_none_forwarded(self) -> None:
        builder = AlertBuilder()
        fp = (1.0, 0.0, 0.0, 0.0)
        evt = builder.build(
            epoch=0,
            fault_posterior=fp,
            entropy_alert=False,
            osnma_alert=False,
            phase_alert=False,
            structure_alert=False,
            failsafe=_make_failsafe(),
            n_active=5,
            mc_auc=None,
        )
        assert evt.mc_auc is None

    def test_mc_auc_float_forwarded(self) -> None:
        builder = AlertBuilder()
        fp = (1.0, 0.0, 0.0, 0.0)
        evt = builder.build(
            epoch=0,
            fault_posterior=fp,
            entropy_alert=False,
            osnma_alert=False,
            phase_alert=False,
            structure_alert=False,
            failsafe=_make_failsafe(),
            n_active=5,
            mc_auc=0.87,
        )
        assert pytest.approx(evt.mc_auc, abs=1e-9) == 0.87

    def test_spoofing_prob_is_fault_posterior_index_3(self) -> None:
        builder = AlertBuilder()
        fp = (0.1, 0.2, 0.3, 0.4)
        evt = builder.build(
            epoch=0,
            fault_posterior=fp,
            entropy_alert=False,
            osnma_alert=False,
            phase_alert=False,
            structure_alert=False,
            failsafe=_make_failsafe(),
            n_active=5,
            mc_auc=None,
        )
        assert pytest.approx(evt.spoofing_prob, abs=1e-9) == 0.4

    def test_failsafe_level_forwarded(self) -> None:
        builder = AlertBuilder()
        fp = (1.0, 0.0, 0.0, 0.0)
        evt = builder.build(
            epoch=0,
            fault_posterior=fp,
            entropy_alert=False,
            osnma_alert=False,
            phase_alert=False,
            structure_alert=False,
            failsafe=_make_failsafe(FailsafeLevel.DEGRADED),
            n_active=5,
            mc_auc=None,
        )
        assert evt.failsafe_level == FailsafeLevel.DEGRADED


# ---------------------------------------------------------------------------
# Tier boundary constants
# ---------------------------------------------------------------------------


class TestTierConstants:
    def test_hard_exclude_above_downweight(self) -> None:
        assert HARD_EXCLUDE_THRESH > DOWNWEIGHT_THRESH

    def test_downweight_thresh_positive(self) -> None:
        assert DOWNWEIGHT_THRESH > 0.0

    def test_hard_exclude_thresh_below_one(self) -> None:
        assert HARD_EXCLUDE_THRESH < 1.0

    def test_spoofing_ins_only_above_degraded(self) -> None:
        assert _SPOOFING_INS_ONLY_THRESH > _SPOOFING_DEGRADED_THRESH

    def test_recovery_epochs_positive(self) -> None:
        assert _FAILSAFE_RECOVERY_EPOCHS >= 1
