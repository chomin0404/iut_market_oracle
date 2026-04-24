"""Matroid combinatorics — log-concavity analysis for portfolio subset selection (T1200).

Background
----------
June Huh (2022 Fields Medal) proved that the characteristic polynomial of any matroid
has log-concave coefficients, resolving the Rota-Heron-Welsh conjecture.

Applied here: the rank-generating polynomial
    P(x) = Σ_k  C(n, k) · alpha^k · beta^(n-k) · x^k
has coefficients b_k = C(n, k) · alpha^k · beta^(n-k) that are log-concave, i.e.
    b_k² ≥ b_{k-1} · b_{k+1}   for all interior k.

Interpretation for portfolio analysis:
    b_k (normalised) ≈ probability that a randomly drawn subset of size k is
    an independent set of the matroid, given rank weight alpha and corank weight beta.
"""

from matroid.log_concavity import compute_log_concave_weights, plot_log_concavity

__all__ = ["compute_log_concave_weights", "plot_log_concavity"]
