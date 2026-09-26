"""Bayesian Bradley-Terry: the core preference model.

Model: P(A preferred over B) = sigmoid(w · (x_A - x_B))
  x = a song's feature vector, w = the listener's taste (one weight per feature).

Bayesian: prior w ~ N(prior_mean, diag(prior_var)). After observing pairs, the
posterior is approximated by a Gaussian (Laplace approximation):
  mean = the most probable w (found with Newton's method),
  cov  = inverse curvature of the log-posterior at that point.
The uncertainty (cov) is what later drives exploration (Thompson sampling).
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, log_expit


@dataclass(frozen=True)
class Posterior:
    mean: np.ndarray  # (d,)
    cov: np.ndarray  # (d, d)

    def utility(self, x: np.ndarray) -> np.ndarray:
        """Expected score of each song (higher = more liked). x: (n, d)."""
        return x @ self.mean

    def prob_prefers(self, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        """P(a preferred over b), averaged over the uncertainty in w.

        Uses the standard probit approximation: sigmoid(mu / sqrt(1 + pi*var/8)),
        which pulls predictions toward 0.5 when the model is unsure.
        """
        diff = np.atleast_2d(x_a - x_b)
        mu = diff @ self.mean
        var = np.einsum("nd,de,ne->n", diff, self.cov, diff)
        return expit(mu / np.sqrt(1 + np.pi * var / 8))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One plausible taste vector (for Thompson sampling)."""
        return rng.multivariate_normal(self.mean, self.cov, method="cholesky")


def fit(
    diffs: np.ndarray,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Posterior:
    """Laplace-approximate posterior from pairwise preferences.

    diffs: (n_pairs, d), each row = x_winner - x_loser.
    prior_var: (d,) prior variance per weight (small = "assume this barely matters").
    """
    d = prior_mean.shape[0]
    if diffs.ndim != 2 or diffs.shape[1] != d:
        raise ValueError(f"diffs must be (n, {d}), got {diffs.shape}")
    if prior_var.shape != (d,) or np.any(prior_var <= 0):
        raise ValueError("prior_var must be a positive vector of length d")
    if not np.isfinite(diffs).all():
        raise ValueError("diffs contain NaN or inf")

    precision = 1.0 / prior_var
    w = prior_mean.astype(float).copy()
    for _ in range(max_iter):
        p = expit(diffs @ w)  # P(winner wins) under current w
        grad = diffs.T @ (1 - p) - precision * (w - prior_mean)  # of log-posterior
        hessian = (diffs.T * (p * (1 - p))) @ diffs + np.diag(precision)
        step = cho_solve(cho_factor(hessian), grad)
        w = _damped(w, step, diffs, prior_mean, precision)
        if np.linalg.norm(step) < tol:
            break

    p = expit(diffs @ w)
    hessian = (diffs.T * (p * (1 - p))) @ diffs + np.diag(precision)
    cov = cho_solve(cho_factor(hessian), np.eye(d))
    return Posterior(mean=w, cov=(cov + cov.T) / 2)


def log_posterior(w, diffs, prior_mean, precision) -> float:
    return float(log_expit(diffs @ w).sum() - 0.5 * np.sum(precision * (w - prior_mean) ** 2))


def _damped(w, step, diffs, prior_mean, precision) -> np.ndarray:
    """Newton step with backtracking, so the fit never gets worse (robust when
    pairs are perfectly separable or the prior is weak)."""
    current = log_posterior(w, diffs, prior_mean, precision)
    scale = 1.0
    for _ in range(30):
        candidate = w + scale * step
        if log_posterior(candidate, diffs, prior_mean, precision) >= current - 1e-12:
            return candidate
        scale /= 2
    return w
