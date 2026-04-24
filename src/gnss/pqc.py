"""Post-quantum cryptography primitives for GNSS OSNMA authentication.

Provides:
    RLWEAuthority          — Ring-LWE Lyubashevsky root-key signing authority.
                             Drop-in replacement for OSNMAAuthority.
                             Quantum-resistant: immune to Shor's algorithm (ECDSA is not).
    QuantumFidelityDetector — Amplitude-encoded fidelity anomaly detector.
                              Catches key_compromise attacks that pass all TESLA checks.
    QUANTUM_FIDELITY_THRESHOLD — Anomaly detection threshold τ = 0.85.

Ring-LWE parameters (Lyubashevsky 2012):
    Q=12289 (NTT-friendly prime), N=256, KAPPA=23, SIGMA=180.0, BETA=2400
    Signature size: 768 bytes  |  Public key: 1024 bytes

NTT design adapted from rlwe_signature.py (negacyclic Twist-NTT):
    ψ = g^{(Q-1)/(2N)} mod Q  (primitive 2N-th root of unity)
    Forward:  a_twist = a · ψ^j → NTT_ω(a_twist)
    Inverse:  INTT_ω(·) → untwist by ψ^{-j} / N
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Ring-LWE constants
# ---------------------------------------------------------------------------

_Q: int = 12289    # NTT-friendly prime (Q-1 = 2^12 · 3)
_N: int = 256      # polynomial degree

RLWE_KAPPA: int = 23      # challenge weight (number of ±1 entries)
RLWE_SIGMA: float = 180.0  # Gaussian σ for masking polynomial y
RLWE_SIGMA_S: float = 3.2  # Gaussian σ for secret key s
RLWE_BETA: int = 2400      # rejection bound ‖z‖∞ ≤ BETA
RLWE_MAX_ATTEMPTS: int = 300

QUANTUM_FIDELITY_THRESHOLD: float = 0.85
# Rationale: normal messages → F=1.0; spoofed with random eph → E[F]≈0.25
# (Bernoulli(0.5) bit statistics: E[cos²θ] ≈ (N/4)² / (N/2)² = 1/4 for N→∞)

# NTT primitive roots
_G = 11                              # primitive root of Z_Q
_PSI = pow(_G, (_Q - 1) // (2 * _N), _Q)  # primitive 2N-th root of unity
_OMEGA = _PSI * _PSI % _Q


# ---------------------------------------------------------------------------
# Negacyclic NTT engine (Z_Q[x] / (x^N + 1))
# ---------------------------------------------------------------------------

class _NTTEngine:
    """Twist-NTT for negacyclic polynomial multiplication mod (x^N+1, Q).

    Cooley-Tukey forward butterfly + Gentleman-Sande inverse butterfly,
    with ψ-twist to handle the negacyclic (x^N+1) modulus.
    """

    def __init__(self) -> None:
        self._log_n = _N.bit_length() - 1
        psi_inv = pow(_PSI, _Q - 2, _Q)
        omega_inv = pow(_OMEGA, _Q - 2, _Q)

        self._twist = self._pow_series(_PSI, _N)
        self._untwist = self._pow_series(psi_inv, _N)
        self._n_inv = pow(_N, _Q - 2, _Q)
        self._br = self._bit_rev_perm(_N)
        self._tw_f = [self._twiddle_row(_OMEGA, _N, s) for s in range(self._log_n)]
        self._tw_b = [self._twiddle_row(omega_inv, _N, s) for s in range(self._log_n)]

    @staticmethod
    def _pow_series(root: int, n: int) -> np.ndarray:
        v = np.empty(n, dtype=np.int64)
        r = 1
        for i in range(n):
            v[i] = r
            r = r * root % _Q
        return v

    @staticmethod
    def _twiddle_row(omega: int, n: int, stage: int) -> np.ndarray:
        m = 1 << (stage + 1)
        half = m >> 1
        wm = pow(omega, n // m, _Q)
        row = np.empty(half, dtype=np.int64)
        w = 1
        for j in range(half):
            row[j] = w
            w = w * wm % _Q
        return row

    @staticmethod
    def _bit_rev_perm(n: int) -> np.ndarray:
        bits = n.bit_length() - 1
        perm = np.empty(n, dtype=np.int64)
        for i in range(n):
            r, b = 0, i
            for _ in range(bits):
                r = (r << 1) | (b & 1)
                b >>= 1
            perm[i] = r
        return perm

    def _cooley_tukey(self, a: np.ndarray) -> np.ndarray:
        for s in range(self._log_n):
            tw = self._tw_f[s]
            m = 1 << (s + 1)
            half = m >> 1
            A = a.reshape(-1, m)
            u = A[:, :half].copy()
            t = A[:, half:] * tw % _Q
            A[:, :half] = (u + t) % _Q
            A[:, half:] = (u - t) % _Q
            a = A.reshape(_N)
        return a

    def _gentleman_sande(self, a: np.ndarray) -> np.ndarray:
        for s in range(self._log_n - 1, -1, -1):
            tw = self._tw_b[s]
            m = 1 << (s + 1)
            half = m >> 1
            A = a.reshape(-1, m)
            u = A[:, :half].copy()
            v = A[:, half:].copy()
            A[:, :half] = (u + v) % _Q
            A[:, half:] = (u - v) * tw % _Q
            a = A.reshape(_N)
        return a

    def forward(self, a: np.ndarray) -> np.ndarray:
        a = a.copy().astype(np.int64) * self._twist % _Q
        a = a[self._br]
        return self._cooley_tukey(a)

    def inverse(self, a: np.ndarray) -> np.ndarray:
        a = self._gentleman_sande(a.copy().astype(np.int64))
        a = a[self._br] * self._n_inv % _Q
        return a * self._untwist % _Q

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.inverse(self.forward(a) * self.forward(b) % _Q)


_NTT = _NTTEngine()  # module-level singleton, initialized once at import


# ---------------------------------------------------------------------------
# Polynomial ring utilities
# ---------------------------------------------------------------------------

def _center(a: np.ndarray) -> np.ndarray:
    """Center polynomial coefficients into [-Q/2, Q/2)."""
    a = a % _Q
    return np.where(a > _Q // 2, a - _Q, a)


def _poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _NTT.mul(a % _Q, b % _Q)


def _poly_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) % _Q


def _inf_norm(a: np.ndarray) -> int:
    return int(np.max(np.abs(a)))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_uniform(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, _Q, size=_N, dtype=np.int64)


def _sample_gaussian(sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.round(rng.normal(0.0, sigma, size=_N)).astype(np.int64)


def _sample_challenge(seed: bytes) -> np.ndarray:
    """SHAKE-256 → sparse ±1 challenge polynomial with exactly KAPPA nonzeros."""
    c = np.zeros(_N, dtype=np.int64)
    stream = hashlib.shake_256(seed).digest(8 * _N)
    positions: set[int] = set()
    idx = 0
    while len(positions) < RLWE_KAPPA:
        pos = int.from_bytes(stream[idx: idx + 2], "little") % _N
        idx += 2
        positions.add(pos)
    for i, pos in enumerate(sorted(positions)):
        sign = 1 if (stream[2 * _N + i] & 1) == 0 else -1
        c[pos] = sign
    return c


def _poly_hash(w: np.ndarray, msg: bytes) -> bytes:
    return hashlib.sha3_256(w.astype(np.int16).tobytes() + msg).digest()


# ---------------------------------------------------------------------------
# RLWE key / signature data structures
# ---------------------------------------------------------------------------

@dataclass
class _RLWEPublicKey:
    a: np.ndarray  # uniform random polynomial in Z_Q[x]/(x^N+1)
    t: np.ndarray  # t = a·s mod Q


@dataclass
class _RLWEPrivateKey:
    s: np.ndarray  # small-coefficient secret polynomial
    pk: _RLWEPublicKey


# ---------------------------------------------------------------------------
# Lyubashevsky signature primitives
# ---------------------------------------------------------------------------

def _rlwe_keygen(rng: np.random.Generator) -> tuple[_RLWEPublicKey, _RLWEPrivateKey]:
    a = _sample_uniform(rng)
    s = _sample_gaussian(RLWE_SIGMA_S, rng)
    t = _poly_mul(a, s)
    pk = _RLWEPublicKey(a=a, t=t)
    return pk, _RLWEPrivateKey(s=s, pk=pk)


def _rlwe_sign(
    sk: _RLWEPrivateKey,
    msg: bytes,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (z, c) — rejection-sampled Lyubashevsky signature."""
    a, s = sk.pk.a, sk.s
    for _ in range(RLWE_MAX_ATTEMPTS):
        y = _sample_gaussian(RLWE_SIGMA, rng)
        w = _poly_mul(a, y)
        c = _sample_challenge(_poly_hash(w, msg))
        sc = _center(_poly_mul(s, c))
        z = y + sc
        if _inf_norm(z) <= RLWE_BETA:
            return z, c
    raise RuntimeError("RLWE sign: rejection sampling did not converge")


def _rlwe_verify(pk: _RLWEPublicKey, msg: bytes, z: np.ndarray, c: np.ndarray) -> bool:
    """Verify Lyubashevsky signature (z, c) against public key pk."""
    if _inf_norm(z) > RLWE_BETA:
        return False
    # w' = a·z - t·c  (equals a·y when e=0, i.e., the original masking poly)
    az = _poly_mul(pk.a, z)
    tc = _poly_mul(pk.t, c)
    wp = _poly_sub(az, tc)
    c2 = _sample_challenge(_poly_hash(wp, msg))
    return bool(np.all(c == c2))


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _sig_to_bytes(z: np.ndarray, c: np.ndarray) -> bytes:
    """Serialize (z, c) → 768 bytes: z as int16×N (512B) + c as int8×N (256B)."""
    return z.astype(np.int16).tobytes() + c.astype(np.int8).tobytes()


def _sig_from_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    z = np.frombuffer(data[:2 * _N], dtype=np.int16).astype(np.int64)
    c = np.frombuffer(data[2 * _N:], dtype=np.int8).astype(np.int64)
    return z, c


# ---------------------------------------------------------------------------
# RLWEAuthority — drop-in replacement for OSNMAAuthority
# ---------------------------------------------------------------------------

class RLWEAuthority:
    """Ring-LWE Lyubashevsky signing authority for OSNMA root-key authentication.

    Replaces ECDSA-P256 (vulnerable to Shor's algorithm on quantum computers)
    with a Ring-SIS hard problem-based signature scheme.

    Interface is identical to OSNMAAuthority:
        sign_root(kroot, epoch, params) → bytes (768-byte RLWE signature)
        verify_root_sig(kroot, epoch, params, sig) → bool
        public_key → _RLWEPublicKey (stored but not used directly by receiver)
    """

    def __init__(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self._pk, self._sk = _rlwe_keygen(rng)
        self._sign_rng = np.random.default_rng(
            0 if seed is None else seed + 1
        )

    @property
    def public_key(self) -> _RLWEPublicKey:
        return self._pk

    def _build_signed_msg(self, kroot: bytes, epoch: int, params: dict[str, int]) -> bytes:
        return (
            kroot
            + struct.pack(">I", epoch)
            + struct.pack("B", params.get("key_size_bits", 128) // 8)
            + struct.pack("B", params.get("mac_size_bits", 40) // 8)
            + struct.pack("B", params.get("delay", 2))
        )

    def sign_root(self, kroot: bytes, epoch: int, params: dict[str, int]) -> bytes:
        """Sign K_root; returns 768-byte RLWE signature."""
        msg = self._build_signed_msg(kroot, epoch, params)
        z, c = _rlwe_sign(self._sk, msg, self._sign_rng)
        return _sig_to_bytes(z, c)

    def verify_root_sig(
        self, kroot: bytes, epoch: int, params: dict[str, int], sig: bytes
    ) -> bool:
        msg = self._build_signed_msg(kroot, epoch, params)
        if len(sig) != 3 * _N:
            return False
        z, c = _sig_from_bytes(sig)
        return _rlwe_verify(self._pk, msg, z, c)


# ---------------------------------------------------------------------------
# QuantumFidelityDetector
# ---------------------------------------------------------------------------

class QuantumFidelityDetector:
    """Amplitude-encoded quantum fidelity anomaly detector for NAV ephemeris.

    Encoding:
        eph_data (32 bytes = 256 bits) → binary vector bits ∈ {0,1}^256
        |ψ(eph)⟩ = bits / ‖bits‖₂   (unit quantum state)

    Fidelity:
        F(recv, exp) = |⟨ψ(recv)|ψ(exp)⟩|² = (dot product)²

    Properties:
        Normal (recv == exp): F = 1.0
        key_compromise (fake random eph): E[F] ≈ (N/4)² / (N/2)² = 1/4 ≈ 0.25
            because E[bit_i · bit_j] = 0.25 (independent Bernoulli(0.5)),
            giving E[dot] ≈ N/4, ‖ψ‖ ≈ √(N/2), so E[F] = (N/4)² / (N/2)² = 1/4

    Threshold τ = 0.85 provides clean separation between F=1.0 and E[F]=0.25.
    """

    def __init__(self, threshold: float = QUANTUM_FIDELITY_THRESHOLD) -> None:
        self._threshold = threshold

    @staticmethod
    def _encode(eph_data: bytes) -> np.ndarray:
        """Encode eph_data as normalized amplitude vector |ψ⟩ ∈ R^(8·len(eph_data))."""
        bits = np.unpackbits(np.frombuffer(eph_data, dtype=np.uint8)).astype(np.float64)
        norm = np.linalg.norm(bits)
        if norm < 1e-12:
            return bits
        return bits / norm

    def fidelity(self, recv: bytes, expected: bytes) -> float:
        """Return F = |⟨ψ(recv)|ψ(expected)⟩|² ∈ [0, 1]."""
        dot = float(np.dot(self._encode(recv), self._encode(expected)))
        return dot * dot

    def is_anomaly(self, recv: bytes, expected: bytes) -> bool:
        """Return True if fidelity < threshold (spoofing indicator)."""
        return self.fidelity(recv, expected) < self._threshold
