"""Bayesian update and MCMC sampling endpoints."""

from __future__ import annotations

import asyncio
from typing import Annotated

import numpy as np
import scipy.linalg as la
from fastapi import APIRouter, Body, HTTPException

from api.schemas.bayesian import (
    BNInferenceRequest,
    BNInferenceResponse,
    ConvergenceDiagnostics,
    HMCRequest,
    MCMCSamplesResponse,
    MHRequest,
    MultivariateNormalTargetSpec,
    NormalTargetSpec,
    TraceSummary,
    UpdateRequest,
    build_network,
)
from bayesian.hmc import run_hmc
from bayesian.kernels import GaussianRWKernel
from bayesian.mh import run_mh
from bayesian.sampler import TargetDistribution
from bayesian.updater import update
from schemas import PosteriorSummary

router = APIRouter()


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------


def _ess(samples: np.ndarray) -> list[float]:
    """Estimate ESS per dimension using Geyer's monotone positive-sequence estimator.

    Parameters
    ----------
    samples:
        Chain of shape ``(n, d)``.

    Returns
    -------
    list[float]
        ESS for each of the ``d`` dimensions, clamped to ``[1, n]``.
    """
    n, d = samples.shape
    result: list[float] = []
    for k in range(d):
        x = samples[:, k]
        x = x - x.mean()
        # Full autocorrelation via FFT
        fft = np.fft.rfft(x, n=2 * n)
        acov = np.fft.irfft(fft * np.conj(fft))[:n].real
        if acov[0] <= 0.0:
            result.append(1.0)
            continue
        acov /= acov[0]  # normalise to autocorrelation
        # Geyer: sum pairs Γ_m = ρ_{2m} + ρ_{2m+1}, stop when Γ_m < 0
        rho_sum = 1.0
        for m in range(1, n // 2):
            gamma = acov[2 * m] + acov[2 * m + 1]
            if gamma < 0.0:
                break
            rho_sum += 2.0 * gamma
        ess_k = float(np.clip(n / rho_sum, 1.0, float(n)))
        result.append(ess_k)
    return result


def _r_hat(samples: np.ndarray) -> list[float]:
    """Split-R-hat per dimension (Vehtari et al. 2021).

    The chain is split into two equal halves which are treated as two
    pseudo-chains of length N = n // 2.  R-hat = sqrt(V-hat / W) where
    W is the within-chain variance and V-hat is a pooled variance estimate.

    Parameters
    ----------
    samples:
        Chain of shape ``(n, d)``.

    Returns
    -------
    list[float]
        R-hat for each of the ``d`` dimensions. Returns 1.0 when the chain
        is too short (n < 4) or the within-chain variance is zero.
    """
    n, d = samples.shape
    result: list[float] = []
    half = n // 2
    if half < 2:
        return [1.0] * d
    chain1 = samples[:half]
    chain2 = samples[half : half * 2]
    for k in range(d):
        c1 = chain1[:, k]
        c2 = chain2[:, k]
        # Within-chain variance (average of per-chain sample variances)
        w = float((np.var(c1, ddof=1) + np.var(c2, ddof=1)) / 2.0)
        if w <= 0.0:
            result.append(1.0)
            continue
        # Between-chain variance: B = N * Var(chain means)
        b = float(half * np.var([c1.mean(), c2.mean()], ddof=1))
        v_hat = (half - 1) / half * w + b / half
        result.append(float(np.sqrt(v_hat / w)))
    return result


def _trace_summary(samples: np.ndarray) -> TraceSummary:
    """Compute per-dimension descriptive statistics.

    Parameters
    ----------
    samples:
        Chain of shape ``(n, d)``.
    """
    percentiles = np.percentile(samples, [2.5, 25.0, 50.0, 75.0, 97.5], axis=0)
    return TraceSummary(
        mean=samples.mean(axis=0).tolist(),
        std=samples.std(axis=0, ddof=1).tolist(),
        q2_5=percentiles[0].tolist(),
        q25=percentiles[1].tolist(),
        q50=percentiles[2].tolist(),
        q75=percentiles[3].tolist(),
        q97_5=percentiles[4].tolist(),
    )


def _convergence_diagnostics(samples: np.ndarray) -> ConvergenceDiagnostics:
    """Build full convergence diagnostics for a post-burn-in chain."""
    return ConvergenceDiagnostics(
        ess=_ess(samples),
        r_hat=_r_hat(samples),
        trace_summary=_trace_summary(samples),
    )


# ---------------------------------------------------------------------------
# Internal: API-side target distribution (isotropic Gaussian)
# ---------------------------------------------------------------------------


class _IsotropicGaussian(TargetDistribution):
    """N(mu, sigma^2 I) target for MCMC endpoints."""

    def __init__(self, mu: np.ndarray, sigma: float) -> None:
        self._mu = mu
        self._sigma = sigma

    @property
    def dim(self) -> int:
        return self._mu.size

    def log_prob(self, x: np.ndarray) -> float:
        delta = x - self._mu
        return float(-0.5 * np.dot(delta, delta) / self._sigma**2)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        return -(x - self._mu) / self._sigma**2


class _MultivariateGaussian(TargetDistribution):
    """N(mu, Sigma) target with full covariance for MCMC endpoints."""

    def __init__(self, mu: np.ndarray, cov: np.ndarray) -> None:
        self._mu = mu
        self._chol = la.cholesky(cov, lower=True)  # Sigma = L Lᵀ

    @property
    def dim(self) -> int:
        return self._mu.size

    def log_prob(self, x: np.ndarray) -> float:
        delta = x - self._mu
        y = la.solve_triangular(self._chol, delta, lower=True)
        d = float(self._mu.size)
        log_det = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
        return float(-0.5 * (d * np.log(2.0 * np.pi) + log_det + np.dot(y, y)))

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        # -Sigma^{-1}(x - mu) = -L^{-T} L^{-1} (x - mu)
        delta = x - self._mu
        y = la.solve_triangular(self._chol, delta, lower=True)
        return -la.solve_triangular(self._chol, y, lower=True, trans="T")


def _build_target(
    spec: NormalTargetSpec | MultivariateNormalTargetSpec,
) -> TargetDistribution:
    if spec.type == "normal":
        return _IsotropicGaussian(mu=np.array(spec.mu), sigma=spec.sigma)
    return _MultivariateGaussian(mu=np.array(spec.mu), cov=np.array(spec.cov))



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


@router.post("/update", response_model=PosteriorSummary)
async def bayesian_update(
    req: Annotated[UpdateRequest, Body(openapi_examples=_BAYESIAN_EXAMPLES)],
) -> PosteriorSummary:
    """Bayesian conjugate update (beta or normal) given a prior and evidence list."""
    try:
        return await asyncio.to_thread(update, req.prior, req.evidence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# MH sampler
# ---------------------------------------------------------------------------

_MH_EXAMPLES = {
    "isotropic_gaussian": {
        "summary": "Isotropic Gaussian N(3, 1)",
        "value": {
            "target": {"type": "normal", "mu": [3.0], "sigma": 1.0},
            "step_size": 0.5,
            "initial": [0.0],
            "n_samples": 500,
            "seed": 42,
            "burn_in": 200,
            "thin": 1,
        },
    },
    "full_cov_gaussian": {
        "summary": "Full-covariance Gaussian N([0,0], [[2,1],[1,2]])",
        "value": {
            "target": {
                "type": "multivariate_normal",
                "mu": [0.0, 0.0],
                "cov": [[2.0, 1.0], [1.0, 2.0]],
            },
            "step_size": 0.8,
            "initial": [0.0, 0.0],
            "n_samples": 500,
            "seed": 0,
            "burn_in": 100,
            "thin": 1,
        },
    },
}


@router.post("/mh/sample", response_model=MCMCSamplesResponse)
async def mh_sample(
    req: Annotated[MHRequest, Body(openapi_examples=_MH_EXAMPLES)],
) -> MCMCSamplesResponse:
    """Run Metropolis–Hastings with a Gaussian random-walk kernel.

    Target distribution is an isotropic or full-covariance Gaussian.
    """
    def _run() -> MCMCSamplesResponse:
        target = _build_target(req.target)
        kernel = GaussianRWKernel(step_size=req.step_size)
        result = run_mh(
            target=target,
            kernel=kernel,
            initial=np.array(req.initial),
            n_samples=req.n_samples,
            rng=np.random.default_rng(req.seed),
            burn_in=req.burn_in,
            thin=req.thin,
        )
        return MCMCSamplesResponse(
            samples=result.samples.tolist(),
            acceptance_rate=result.acceptance_rate,
            n_accepted=result.n_accepted,
            n_total=result.n_total,
            diagnostics=_convergence_diagnostics(result.samples),
        )

    try:
        return await asyncio.to_thread(_run)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid sampler parameters.")


# ---------------------------------------------------------------------------
# HMC sampler
# ---------------------------------------------------------------------------


_HMC_EXAMPLES = {
    "isotropic_gaussian": {
        "summary": "Isotropic Gaussian N(0, 1)",
        "value": {
            "target": {"type": "normal", "mu": [0.0], "sigma": 1.0},
            "step_size": 0.3,
            "n_leapfrog": 10,
            "initial": [0.0],
            "n_samples": 500,
            "seed": 42,
            "burn_in": 100,
        },
    },
    "full_cov_gaussian": {
        "summary": "Full-covariance Gaussian N([1,2], [[2,1],[1,2]])",
        "value": {
            "target": {
                "type": "multivariate_normal",
                "mu": [1.0, 2.0],
                "cov": [[2.0, 1.0], [1.0, 2.0]],
            },
            "step_size": 0.3,
            "n_leapfrog": 15,
            "initial": [0.0, 0.0],
            "n_samples": 500,
            "seed": 0,
            "burn_in": 100,
            "mass": [1.0, 1.0],
        },
    },
}


@router.post("/hmc/sample", response_model=MCMCSamplesResponse)
async def hmc_sample(
    req: Annotated[HMCRequest, Body(openapi_examples=_HMC_EXAMPLES)],
) -> MCMCSamplesResponse:
    """Run Hamiltonian Monte Carlo.

    Target distribution is an isotropic or full-covariance Gaussian.
    Requires an analytic gradient — provided internally for this target.
    """
    def _run() -> MCMCSamplesResponse:
        target = _build_target(req.target)
        result = run_hmc(
            target=target,
            step_size=req.step_size,
            n_leapfrog=req.n_leapfrog,
            initial=np.array(req.initial),
            n_samples=req.n_samples,
            rng=np.random.default_rng(req.seed),
            burn_in=req.burn_in,
            thin=req.thin,
            mass=np.array(req.mass) if req.mass is not None else None,
        )
        return MCMCSamplesResponse(
            samples=result.samples.tolist(),
            acceptance_rate=result.acceptance_rate,
            n_accepted=result.n_accepted,
            n_total=result.n_total,
            diagnostics=_convergence_diagnostics(result.samples),
        )

    try:
        return await asyncio.to_thread(_run)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid sampler parameters.")


# ---------------------------------------------------------------------------
# Bayesian Network inference
# ---------------------------------------------------------------------------

_BN_EXAMPLES = {
    "market_regime": {
        "summary": "Economy -> Regime (evidence ari)",
        "value": {
            "network": {
                "nodes": [
                    {"node_id": "economy", "states": ["expansion", "recession"]},
                    {"node_id": "regime",  "states": ["bull", "bear", "neutral"]},
                ],
                "edges": [{"parent": "economy", "child": "regime"}],
                "cpds": [
                    {"node_id": "economy", "probs": [0.7, 0.3]},
                    {
                        "node_id": "regime",
                        "rows": [
                            {"parent_states": ["expansion"], "probs": [0.6, 0.1, 0.3]},
                            {"parent_states": ["recession"], "probs": [0.2, 0.6, 0.2]},
                        ],
                    },
                ],
            },
            "evidence": {"economy": "expansion"},
            "queries": ["regime"],
        },
    },
    "no_evidence": {
        "summary": "Prior marginal (evidence nashi)",
        "value": {
            "network": {
                "nodes": [
                    {"node_id": "economy", "states": ["expansion", "recession"]},
                    {"node_id": "regime",  "states": ["bull", "bear", "neutral"]},
                ],
                "edges": [{"parent": "economy", "child": "regime"}],
                "cpds": [
                    {"node_id": "economy", "probs": [0.7, 0.3]},
                    {
                        "node_id": "regime",
                        "rows": [
                            {"parent_states": ["expansion"], "probs": [0.6, 0.1, 0.3]},
                            {"parent_states": ["recession"], "probs": [0.2, 0.6, 0.2]},
                        ],
                    },
                ],
            },
            "evidence": {},
            "queries": ["economy", "regime"],
        },
    },
}


@router.post("/network/infer", response_model=BNInferenceResponse)
async def network_infer(
    req: Annotated[BNInferenceRequest, Body(openapi_examples=_BN_EXAMPLES)],
) -> BNInferenceResponse:
    """Exact inference on a discrete Bayesian Network via Variable Elimination.

    Builds the network from the supplied spec, applies optional evidence,
    and returns posterior distributions P(query | evidence) for every
    requested query node.

    **Evidence** is optional. Omit or pass ``{}`` for prior marginals.
    Querying an observed node returns a degenerate distribution
    (probability 1 on the observed state).
    """
    def _run() -> BNInferenceResponse:
        net = build_network(req.network)
        try:
            for node_id, state in req.evidence.items():
                net.observe(node_id, state)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            posteriors = {q: net.posterior(q) for q in req.queries}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return BNInferenceResponse(posteriors=posteriors)

    return await asyncio.to_thread(_run)
