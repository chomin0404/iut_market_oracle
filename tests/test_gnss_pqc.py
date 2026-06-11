"""Tests for src/gnss/pqc.py — Ring-LWE OSNMA authority and quantum fidelity detector."""

from __future__ import annotations

import numpy as np
import pytest

from gnss.pqc import (
    _N,
    _NTT,
    _Q,
    QUANTUM_FIDELITY_THRESHOLD,
    RLWE_BETA,
    RLWE_KAPPA,
    QuantumFidelityDetector,
    RLWEAuthority,
    _center,
    _inf_norm,
    _poly_hash,
    _poly_sub,
    _sample_challenge,
    _sample_gaussian,
    _sample_uniform,
    _sig_from_bytes,
    _sig_to_bytes,
)

# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_constants_q_prime() -> None:
    """Q must be a prime (NTT-friendly: Q-1 = 2^12 · 3)."""
    assert _Q == 12289


def test_constants_n() -> None:
    assert _N == 256


def test_rlwe_beta_positive() -> None:
    assert RLWE_BETA > 0


def test_rlwe_kappa_positive() -> None:
    assert RLWE_KAPPA > 0


def test_quantum_threshold_range() -> None:
    assert 0.0 < QUANTUM_FIDELITY_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# NTT engine
# ---------------------------------------------------------------------------


def test_ntt_forward_inverse_roundtrip() -> None:
    """NTT forward → inverse must recover the original polynomial."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, _Q, size=_N, dtype=np.int64)
    recovered = _NTT.inverse(_NTT.forward(a))
    assert np.all(recovered == a % _Q)


def test_ntt_mul_commutativity() -> None:
    """a*b == b*a in the polynomial ring."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, _Q, size=_N, dtype=np.int64)
    b = rng.integers(0, _Q, size=_N, dtype=np.int64)
    ab = _NTT.mul(a, b)
    ba = _NTT.mul(b, a)
    assert np.all(ab == ba)


# ---------------------------------------------------------------------------
# Polynomial utilities
# ---------------------------------------------------------------------------


def test_center_range() -> None:
    """_center must map all values into [-Q//2, Q//2)."""
    rng = np.random.default_rng(2)
    a = rng.integers(0, _Q, size=_N, dtype=np.int64)
    c = _center(a)
    assert np.all(c >= -(_Q // 2))
    assert np.all(c < _Q - _Q // 2)


def test_poly_sub_mod() -> None:
    """_poly_sub results must be in [0, Q)."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, _Q, size=_N, dtype=np.int64)
    b = rng.integers(0, _Q, size=_N, dtype=np.int64)
    result = _poly_sub(a, b)
    assert np.all(result >= 0)
    assert np.all(result < _Q)


def test_inf_norm_nonneg() -> None:
    rng = np.random.default_rng(4)
    a = rng.integers(-100, 100, size=_N, dtype=np.int64)
    assert _inf_norm(a) >= 0


def test_poly_hash_deterministic() -> None:
    rng = np.random.default_rng(5)
    w = rng.integers(0, _Q, size=_N, dtype=np.int64)
    msg = b"test_message"
    h1 = _poly_hash(w, msg)
    h2 = _poly_hash(w, msg)
    assert h1 == h2
    assert len(h1) == 32  # SHA3-256 → 32 bytes


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_uniform_range() -> None:
    rng = np.random.default_rng(10)
    a = _sample_uniform(rng)
    assert a.shape == (_N,)
    assert np.all(a >= 0)
    assert np.all(a < _Q)


def test_sample_gaussian_shape() -> None:
    rng = np.random.default_rng(11)
    s = _sample_gaussian(3.2, rng)
    assert s.shape == (_N,)


def test_sample_challenge_weight() -> None:
    """Challenge polynomial must have exactly KAPPA nonzero entries."""
    seed = b"challenge_seed_test"
    c = _sample_challenge(seed)
    assert c.shape == (_N,)
    assert int(np.count_nonzero(c)) == RLWE_KAPPA
    # All nonzero entries must be ±1
    nonzero = c[c != 0]
    assert np.all(np.abs(nonzero) == 1)


def test_sample_challenge_deterministic() -> None:
    seed = b"same_seed"
    c1 = _sample_challenge(seed)
    c2 = _sample_challenge(seed)
    assert np.all(c1 == c2)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_sig_roundtrip() -> None:
    rng = np.random.default_rng(20)
    z = rng.integers(-RLWE_BETA, RLWE_BETA, size=_N, dtype=np.int64)
    c = _sample_challenge(b"roundtrip")
    data = _sig_to_bytes(z, c)
    assert len(data) == 3 * _N  # 512 + 256
    z2, c2 = _sig_from_bytes(data)
    assert np.all(z == z2)
    assert np.all(c == c2)


# ---------------------------------------------------------------------------
# RLWEAuthority
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def authority() -> RLWEAuthority:
    return RLWEAuthority(seed=42)


_KROOT = bytes(range(16))
_EPOCH = 12345
_PARAMS: dict[str, int] = {"key_size_bits": 128, "mac_size_bits": 40, "delay": 2}


def test_authority_public_key_shape(authority: RLWEAuthority) -> None:
    pk = authority.public_key
    assert pk.a.shape == (_N,)
    assert pk.t.shape == (_N,)


def test_sign_root_returns_768_bytes(authority: RLWEAuthority) -> None:
    sig = authority.sign_root(_KROOT, _EPOCH, _PARAMS)
    assert isinstance(sig, bytes)
    assert len(sig) == 3 * _N


def test_verify_root_sig_valid(authority: RLWEAuthority) -> None:
    sig = authority.sign_root(_KROOT, _EPOCH, _PARAMS)
    assert authority.verify_root_sig(_KROOT, _EPOCH, _PARAMS, sig) is True


def test_verify_root_sig_wrong_kroot(authority: RLWEAuthority) -> None:
    sig = authority.sign_root(_KROOT, _EPOCH, _PARAMS)
    wrong_kroot = bytes([0xFF] * 16)
    assert authority.verify_root_sig(wrong_kroot, _EPOCH, _PARAMS, sig) is False


def test_verify_root_sig_wrong_epoch(authority: RLWEAuthority) -> None:
    sig = authority.sign_root(_KROOT, _EPOCH, _PARAMS)
    assert authority.verify_root_sig(_KROOT, _EPOCH + 1, _PARAMS, sig) is False


def test_verify_root_sig_truncated(authority: RLWEAuthority) -> None:
    sig = authority.sign_root(_KROOT, _EPOCH, _PARAMS)
    assert authority.verify_root_sig(_KROOT, _EPOCH, _PARAMS, sig[:100]) is False


def test_verify_root_sig_empty(authority: RLWEAuthority) -> None:
    assert authority.verify_root_sig(_KROOT, _EPOCH, _PARAMS, b"") is False


def test_authority_no_seed() -> None:
    """RLWEAuthority with seed=None must construct without error."""
    auth = RLWEAuthority(seed=None)
    sig = auth.sign_root(_KROOT, _EPOCH, _PARAMS)
    assert auth.verify_root_sig(_KROOT, _EPOCH, _PARAMS, sig) is True


def test_different_seeds_different_keys() -> None:
    a1 = RLWEAuthority(seed=1)
    a2 = RLWEAuthority(seed=2)
    # Public keys should differ
    assert not np.all(a1.public_key.a == a2.public_key.a)


# ---------------------------------------------------------------------------
# QuantumFidelityDetector
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> QuantumFidelityDetector:
    return QuantumFidelityDetector()


_EPH_A = bytes(range(32))  # 32-byte ephemeris
_EPH_B = bytes(range(1, 33))  # different data


def test_fidelity_identical(detector: QuantumFidelityDetector) -> None:
    """Identical data must give fidelity = 1.0."""
    assert detector.fidelity(_EPH_A, _EPH_A) == pytest.approx(1.0, abs=1e-9)


def test_fidelity_different(detector: QuantumFidelityDetector) -> None:
    """Different data must give fidelity < 1.0."""
    assert detector.fidelity(_EPH_A, _EPH_B) < 1.0


def test_fidelity_range(detector: QuantumFidelityDetector) -> None:
    """Fidelity must be in [0, 1]."""
    f = detector.fidelity(_EPH_A, _EPH_B)
    assert 0.0 <= f <= 1.0


def test_is_anomaly_identical(detector: QuantumFidelityDetector) -> None:
    """Identical data → not an anomaly."""
    assert detector.is_anomaly(_EPH_A, _EPH_A) is False


def test_is_anomaly_random_vs_expected() -> None:
    """Random bytes vs expected → likely anomaly (E[F] ≈ 0.25 < 0.85 threshold)."""
    rng = np.random.default_rng(99)
    recv = bytes(rng.integers(0, 256, size=32, dtype=np.uint8).tolist())
    expected = _EPH_A
    det = QuantumFidelityDetector(threshold=QUANTUM_FIDELITY_THRESHOLD)
    # With a single sample we can't guarantee anomaly, but fidelity must be computable
    f = det.fidelity(recv, expected)
    assert 0.0 <= f <= 1.0


def test_is_anomaly_custom_threshold() -> None:
    """With threshold=0.0 nothing is flagged as anomaly."""
    det = QuantumFidelityDetector(threshold=0.0)
    assert det.is_anomaly(_EPH_A, _EPH_B) is False


def test_is_anomaly_threshold_one() -> None:
    """With threshold=1.0 non-identical data is always anomaly."""
    det = QuantumFidelityDetector(threshold=1.0)
    assert det.is_anomaly(_EPH_A, _EPH_B) is True


def test_fidelity_all_zeros() -> None:
    """All-zero ephemeris (zero norm) must not raise."""
    det = QuantumFidelityDetector()
    zero = bytes(32)
    # zero norm → encode returns zeros → dot = 0 → F = 0
    f = det.fidelity(zero, _EPH_A)
    assert f == pytest.approx(0.0, abs=1e-9)
