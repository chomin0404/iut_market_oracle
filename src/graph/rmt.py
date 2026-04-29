"""Random Matrix Theory denoising via Marchenko-Pastur law (Tao-Vu 2011).

Mathematical basis
------------------
Marchenko-Pastur law (1967)
    For a p × n data matrix X with i.i.d. zero-mean variance-σ² entries,
    the empirical spectral distribution of the sample covariance
    Ĉ = (1/n) X Xᵀ converges almost surely to the MP distribution as
    p, n → ∞ with aspect ratio q = p/n → c > 0:

        ρ_MP(λ; σ², q) = √((λ_+ − λ)(λ − λ_−)) / (2π q σ² λ)
                         for λ ∈ [λ_−, λ_+]

    Noise eigenvalue bounds:
        λ_± = σ²(1 ± √q)²

    When q > 1 the spectrum has a point mass (1 − 1/q) at λ = 0 in addition
    to the continuous bulk on [λ_−, λ_+] with λ_− = σ²(√q − 1)² > 0.

Tao-Vu universality (2011)
    The bulk and edge eigenvalue statistics of a Wigner (Hermitian random)
    matrix are universal: they converge to GUE statistics regardless of the
    entry distribution, provided the entries have finite (4+ε)-th moments.
    As a corollary, the MP law applies to sample covariance matrices of
    non-Gaussian returns (e.g. Laplace, Student-t) — the standard setting
    in financial applications.

Variance estimation
    For a proper correlation matrix (diagonal = 1):  σ² = trace(C)/p = 1.
    For an un-normalised matrix:  σ² = trace(C)/p   (mean eigenvalue).
    This single formula handles both cases without branching.

Eigenvalue clipping (denoising)
    Eigenvalues λᵢ ≤ λ_+ are noise; replace them with their mean μ_noise
    (preserves the noise sub-trace, hence the total trace after rounding).
    Eigenvalues λᵢ > λ_+ are signal; kept unchanged.
    Reconstruct:  C_clean = V diag(λ_clean) Vᵀ, then re-normalise diagonal.

Denoised dependency concentration
    HHI of the column-sum (off-diagonal absolute values) of C_clean.
    This is the "true" dependency concentration after removing correlations
    attributable to random sampling noise.

Reference
---------
Laloux L., Cizeau P., Bouchaud J.-P., Potters M. (1999).
    "Noise Dressing of Financial Correlation Matrices."
    Physical Review Letters, 83(7), 1467–1470.

Tao T., Vu V. (2011).
    "Random Matrices: Universality of Local Eigenvalue Statistics."
    Acta Mathematica, 206(1), 127–204.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

_DIAG_FLOOR: float = 1e-12  # minimum diagonal value before re-normalisation


@dataclass
class MPBounds:
    """Marchenko-Pastur noise eigenvalue bounds.

    Attributes
    ----------
    lower : float
        λ_− = σ²(1 − √q)² = σ²(√q − 1)² ≥ 0.  The bulk lower edge of the
        continuous MP density.  When q > 1 the spectrum also has a point mass
        at λ = 0 (fraction 1 − 1/q), but that is separate from this bound.
    upper : float
        λ_+ = σ²(1 + √q)².
    sigma_sq : float
        Estimated variance σ² = trace(C) / p.
    q : float
        Aspect ratio q = p / n_obs.
    """

    lower: float
    upper: float
    sigma_sq: float
    q: float


@dataclass
class DenoisingResult:
    """Output of Marchenko-Pastur eigenvalue clipping.

    Attributes
    ----------
    matrix_cleaned : np.ndarray
        p × p cleaned, symmetrised matrix with re-normalised diagonal.
    eigenvalues_raw : list[float]
        Eigenvalues of the input matrix in ascending order.
    eigenvalues_cleaned : list[float]
        Eigenvalues after clipping in ascending order.
    n_signal : int
        Number of eigenvalues strictly above λ_+  (genuine structure).
    n_noise : int
        Number of eigenvalues at or below λ_+  (attributable to noise).
    mp_bounds : MPBounds
        The Marchenko-Pastur bounds used.
    hhi_raw : float
        Herfindahl-Hirschman Index of off-diagonal column-sum shares
        computed on the *raw* input matrix.  Range [0, 1].
    hhi_cleaned : float
        HHI of off-diagonal column-sum shares after denoising.  Range [0, 1].
        A lower value than hhi_raw means spurious concentrations were removed.
    """

    matrix_cleaned: np.ndarray = field(repr=False)
    eigenvalues_raw: list[float]
    eigenvalues_cleaned: list[float]
    n_signal: int
    n_noise: int
    mp_bounds: MPBounds
    hhi_raw: float
    hhi_cleaned: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def marchenko_pastur_bounds(
    n_vars: int,
    n_obs: int,
    sigma_sq: float = 1.0,
) -> MPBounds:
    """Return the Marchenko-Pastur noise eigenvalue bounds.

    Parameters
    ----------
    n_vars : int
        Number of variables p (matrix dimension).  Must be ≥ 1.
    n_obs : int
        Number of observations n used to estimate the matrix.  Must be ≥ 1.
    sigma_sq : float
        Assumed variance σ².  Default 1.0, which is correct for a
        sample *correlation* matrix (diagonal = 1).

    Returns
    -------
    MPBounds
        λ_± = σ²(1 ± √q)², q = p / n.

    Raises
    ------
    ValueError
        If n_vars < 1, n_obs < 1, or sigma_sq ≤ 0.

    Examples
    --------
    q = 1  (p = n):  λ_+ = σ²·4,  λ_- = 0
    q = 0.25 (p = n/4):  λ_+ = σ²·(1+0.5)² = 2.25σ²,  λ_- = σ²·0.25

    Notes
    -----
    When n_obs < n_vars (q > 1) the correlation matrix is rank-deficient
    and λ_- is clamped to 0.  The denoising is still well-defined but
    the signal/noise separation is less reliable.
    """
    if n_vars < 1:
        raise ValueError(f"n_vars must be >= 1, got {n_vars}")
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")
    if sigma_sq <= 0.0:
        raise ValueError(f"sigma_sq must be > 0, got {sigma_sq}")

    q = n_vars / n_obs
    sq = q**0.5
    upper = sigma_sq * (1.0 + sq) ** 2
    lower = sigma_sq * max(0.0, (1.0 - sq) ** 2)
    return MPBounds(lower=lower, upper=upper, sigma_sq=sigma_sq, q=q)


def denoise_correlation_matrix(
    C: np.ndarray,
    n_obs: int,
) -> DenoisingResult:
    """Denoise a symmetric positive-semidefinite matrix via MP eigenvalue clipping.

    Algorithm
    ---------
    1. Eigendecompose C = V Λ Vᵀ  (eigh; ascending eigenvalues).
    2. Estimate σ² = trace(C) / p  (= 1 for a proper correlation matrix).
    3. Compute MP upper bound λ_+ = σ²(1 + √q)².
    4. Classify eigenvalues: signal if λᵢ > λ_+, noise otherwise.
    5. Replace noise eigenvalues with μ_noise = mean(noise eigenvalues).
    6. Reconstruct C_clean = V diag(λ_clean) Vᵀ.
    7. Force symmetry; re-normalise diagonal to the original diagonal values.
    8. Compute HHI on off-diagonal column sums before and after denoising.

    Parameters
    ----------
    C : np.ndarray
        p × p symmetric positive-semidefinite matrix.
        Should be a sample correlation or covariance matrix.
    n_obs : int
        Number of observations n used to estimate C.  Must be ≥ 1.

    Returns
    -------
    DenoisingResult

    Raises
    ------
    ValueError
        If C is not 2-D, not square, not symmetric (tolerance 1e-8),
        or if n_obs < 1.

    Notes
    -----
    Floating-point note: eigh returns real eigenvalues in ascending order.
    Slightly negative eigenvalues (numerical noise from non-PSD inputs)
    are treated as noise-class eigenvalues; after clipping they become
    μ_noise > 0, which restores positive semidefiniteness.
    """
    if C.ndim != 2:
        raise ValueError(f"C must be a 2-D array, got shape {C.shape}")
    if C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square, got shape {C.shape}")
    if not np.allclose(C, C.T, atol=1e-8):
        raise ValueError("C must be symmetric (|C - Cᵀ| > 1e-8 detected)")
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")

    p = C.shape[0]

    # Save original diagonal for re-normalisation
    orig_diag = np.diag(C).copy()

    # --- Step 1: eigendecompose (ascending order) -----------------------
    eigvals, eigvecs = np.linalg.eigh(C)
    # eigh guarantees real eigenvalues for symmetric input

    # --- Step 2: estimate σ² -------------------------------------------
    sigma_sq = float(np.trace(C)) / p  # = 1.0 for correlation matrix

    # --- Step 3: MP bounds ---------------------------------------------
    bounds = marchenko_pastur_bounds(p, n_obs, sigma_sq=sigma_sq)

    # --- Step 4: classify eigenvalues ----------------------------------
    signal_mask = eigvals > bounds.upper  # strictly above λ_+
    noise_mask = ~signal_mask

    # --- Step 5: clip noise eigenvalues --------------------------------
    eigvals_cleaned = eigvals.copy()
    if noise_mask.any():
        mu_noise = float(eigvals[noise_mask].mean())
        eigvals_cleaned[noise_mask] = mu_noise

    # --- Step 6: reconstruct ------------------------------------------
    C_clean = eigvecs @ np.diag(eigvals_cleaned) @ eigvecs.T

    # --- Step 7: force symmetry + re-normalise diagonal ---------------
    C_clean = (C_clean + C_clean.T) * 0.5
    new_diag = np.diag(C_clean)
    # Re-scale each row/col so diagonal matches the original diagonal.
    # For a correlation matrix orig_diag = 1; for covariance it retains σᵢ².
    scale = np.where(new_diag > _DIAG_FLOOR, np.sqrt(orig_diag / new_diag), 0.0)
    C_clean = C_clean * np.outer(scale, scale)

    # --- Step 8: compute HHI before and after -------------------------
    hhi_raw = _hhi_offdiag(C)
    hhi_cleaned = _hhi_offdiag(C_clean)

    return DenoisingResult(
        matrix_cleaned=C_clean,
        eigenvalues_raw=eigvals.tolist(),
        eigenvalues_cleaned=eigvals_cleaned.tolist(),
        n_signal=int(signal_mask.sum()),
        n_noise=int(noise_mask.sum()),
        mp_bounds=bounds,
        hhi_raw=hhi_raw,
        hhi_cleaned=hhi_cleaned,
    )


def rmt_dependency_concentration(
    C: np.ndarray,
    n_obs: int,
) -> float:
    """Dependency concentration (HHI) after Marchenko-Pastur denoising.

    Computes the Herfindahl-Hirschman Index of the off-diagonal column-sum
    shares of the denoised correlation matrix.  Lower values indicate that
    the observed concentration was partially an artefact of finite-sample
    noise; higher values confirm a genuine structural hub.

    Parameters
    ----------
    C : np.ndarray
        p × p symmetric PSD matrix (typically a sample correlation matrix).
    n_obs : int
        Number of observations used to estimate C.

    Returns
    -------
    float
        HHI in [0, 1].  0 when there are no off-diagonal connections;
        1/p when connections are perfectly uniform; approaching 1 when
        one node dominates.

    See Also
    --------
    denoise_correlation_matrix : Full denoising output.
    graph.metrics.dependency_concentration : Raw HHI without denoising.
    """
    return denoise_correlation_matrix(C, n_obs).hhi_cleaned


# ---------------------------------------------------------------------------
# Helper: build symmetric matrix from GraphInput
# ---------------------------------------------------------------------------


def strength_matrix(
    graph: "GraphInput",  # type: ignore[name-defined]  # noqa: F821, UP037
    node_order: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a symmetrised strength matrix from a GraphInput.

    The directed strength matrix A is defined as:
        A[i, j] = edge.strength  if edge source=i, target=j exists (else 0)

    The symmetrised version is:
        S[i, j] = (A[i, j] + A[j, i]) / 2   for i ≠ j
        S[i, i] = 1.0                         (self-correlation proxy)

    The off-diagonal values are divided by the maximum off-diagonal value
    so that S[i, j] ∈ [0, 1] and the matrix qualifies as a correlation
    matrix proxy for MP denoising.

    Parameters
    ----------
    graph : GraphInput
        A validated GraphInput.
    node_order : list[str] | None
        Explicit node ordering.  Defaults to the order in graph.nodes.

    Returns
    -------
    (S, node_order)
        S          : p × p numpy array (symmetric, diagonal = 1).
        node_order : list of node IDs corresponding to row/column indices.

    Notes
    -----
    Floating-point note: if all off-diagonal strengths are zero (no edges),
    the normalisation is skipped and S = I_p.  This is the correct
    degenerate case: all nodes are independent.
    """

    if node_order is None:
        node_order = [n.node_id for n in graph.nodes]

    idx = {nid: i for i, nid in enumerate(node_order)}
    p = len(node_order)
    A = np.zeros((p, p), dtype=float)

    for edge in graph.edges:
        i = idx[edge.source]
        j = idx[edge.target]
        A[i, j] = edge.strength

    # Symmetrise off-diagonal
    S = (A + A.T) * 0.5
    max_val = np.abs(S).max()
    if max_val > 0.0:
        S = S / max_val  # normalise to [0, 1]

    # Set diagonal to 1 (self-correlation = perfect)
    np.fill_diagonal(S, 1.0)

    return S, node_order


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hhi_offdiag(C: np.ndarray) -> float:
    """HHI of off-diagonal column-sum shares of |C|.

    Returns 0.0 when all off-diagonal entries are zero.
    """
    p = C.shape[0]
    # Sum of absolute off-diagonal entries per column
    mask = ~np.eye(p, dtype=bool)
    col_sums = np.abs(C)[mask].reshape(p, p - 1).sum(axis=1)
    total = col_sums.sum()
    if total == 0.0:
        return 0.0
    shares = col_sums / total
    return float((shares**2).sum())
