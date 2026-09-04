"""BOCPD implementation: hazard rate, run-length posterior P(r_t | x_1:t) (Adams & MacKay 2007).

Standard formulation for a real-valued observation stream: constant hazard rate H (so run lengths
are a priori geometric), and a Normal-Inverse-Gamma conjugate predictive model per run-length
hypothesis (Student-t predictive distribution). This is the textbook off-the-shelf BOCPD used for a
scalar signal -- here, the rolling residual-loss signal from `bocpd/signals.py` -- not something
specific to XGBoost; the boosting integration happens in Phase 3, this module only knows about a
1-D real-valued time series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class NormalInverseGammaPrior:
    """Conjugate prior for a Normal observation model with unknown mean and variance."""
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0


class BOCPD:
    """Online run-length posterior tracker.

    Call `update(x)` once per new observation `x`; it returns the current run-length posterior
    `P(r_t = r | x_1:t)` as an array indexed by run length (index 0 = "a changepoint just happened",
    index r = "r steps since the last changepoint"). Internal state grows by one hypothesis per step
    and is pruned once a hypothesis's posterior mass drops below `prune_threshold`, keeping the
    per-step cost roughly bounded rather than growing linearly with total stream length.
    """

    def __init__(
        self,
        hazard_rate: float = 1.0 / 250.0,
        prior: NormalInverseGammaPrior | None = None,
        prune_threshold: float = 1e-8,
    ):
        self.hazard_rate = hazard_rate
        self.prior = prior or NormalInverseGammaPrior()
        self.prune_threshold = prune_threshold
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.run_length_posterior = np.array([1.0])
        self.mu = np.array([self.prior.mu0])
        self.kappa = np.array([self.prior.kappa0])
        self.alpha = np.array([self.prior.alpha0])
        self.beta = np.array([self.prior.beta0])

    def _predictive_pdf(self, x: float) -> np.ndarray:
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        df = 2 * self.alpha
        return stats.t.pdf(x, df=df, loc=self.mu, scale=scale)

    def update(self, x: float) -> np.ndarray:
        pred_probs = self._predictive_pdf(x)

        growth_probs = self.run_length_posterior * pred_probs * (1.0 - self.hazard_rate)
        cp_prob = float(np.sum(self.run_length_posterior * pred_probs * self.hazard_rate))
        new_posterior = np.concatenate(([cp_prob], growth_probs))

        total = new_posterior.sum()
        if total <= 0 or not np.isfinite(total):
            # Numerical underflow (e.g. wildly out-of-distribution x): fall back to a hard reset
            # rather than propagating NaNs through the rest of the run.
            logger.warning("BOCPD: non-finite/zero posterior mass at t=%d, resetting run-length posterior", self.t)
            new_posterior = np.zeros_like(new_posterior)
            new_posterior[0] = 1.0
        else:
            new_posterior /= total

        new_mu = np.concatenate(([self.prior.mu0], (self.kappa * self.mu + x) / (self.kappa + 1)))
        new_kappa = np.concatenate(([self.prior.kappa0], self.kappa + 1))
        new_alpha = np.concatenate(([self.prior.alpha0], self.alpha + 0.5))
        new_beta = np.concatenate((
            [self.prior.beta0],
            self.beta + (self.kappa * (x - self.mu) ** 2) / (2 * (self.kappa + 1)),
        ))

        keep = new_posterior >= self.prune_threshold
        keep[0] = True
        if keep.sum() < len(keep):
            new_posterior = new_posterior[keep]
            new_posterior /= new_posterior.sum()
            new_mu, new_kappa, new_alpha, new_beta = new_mu[keep], new_kappa[keep], new_alpha[keep], new_beta[keep]

        self.run_length_posterior = new_posterior
        self.mu, self.kappa, self.alpha, self.beta = new_mu, new_kappa, new_alpha, new_beta
        self.t += 1

        return self.run_length_posterior.copy()

    def map_run_length(self) -> int:
        return int(np.argmax(self.run_length_posterior))

    def changepoint_probability(self) -> float:
        """P(r_t = 0 | x_1:t).

        NOTE: for a *constant* hazard rate h (used here), this quantity is a mathematical identity
        that equals h at every single timestep, regardless of the data -- provably so, since the
        r_t=0 posterior mass is always exactly h times the same normalizing evidence term that
        produces the rest of the posterior. It is therefore NOT a usable changepoint-detection signal
        on its own (it never deviates from h). It's kept here because it's a well-defined quantity of
        the model, but detection should use `post_break_weight()` instead, which aggregates over a
        small window of low run lengths and genuinely is data-dependent (see that method's docstring).
        """
        return float(self.run_length_posterior[0])

    def post_break_weight(self, break_window: int) -> float:
        """P(r_t < break_window | x_1:t) -- probability the current step is within `break_window`
        steps of the most recent changepoint. Unlike `changepoint_probability()` (pinned to the
        constant hazard rate for the r_t=0 term alone), this aggregates several low run-length
        hypotheses whose individual posterior mass genuinely depends on how well the data matches a
        "freshly reset" model vs. an established one -- empirically this is a real, sharp,
        data-dependent spike right after a true break (see tests/test_bocpd.py), and is the primitive
        Phase 3's soft instance-weighting (Section 3, point 4) is built from: a continuous "how likely
        is this a post-break instance" signal rather than a hard run-length threshold.
        """
        idx = min(break_window, len(self.run_length_posterior))
        return float(self.run_length_posterior[:idx].sum())


def run_bocpd_on_signal(
    signal: np.ndarray,
    hazard_rate: float = 1.0 / 250.0,
    prior: NormalInverseGammaPrior | None = None,
    prune_threshold: float = 1e-8,
    break_window: int = 10,
) -> dict:
    """Bulk convenience wrapper: runs BOCPD over an entire 1-D signal and returns per-timestep
    summaries (MAP run length, raw changepoint probability, and post-break weight -- the actually
    useful detection signal, see `BOCPD.post_break_weight`) plus the detector instance itself (whose
    final `run_length_posterior` reflects the last timestep only -- use the returned arrays for the
    full trajectory).
    """
    detector = BOCPD(hazard_rate=hazard_rate, prior=prior, prune_threshold=prune_threshold)
    n = len(signal)
    map_run_length = np.empty(n, dtype=int)
    changepoint_prob = np.empty(n, dtype=float)
    post_break_weight = np.empty(n, dtype=float)

    for t, x in enumerate(signal):
        detector.update(float(x))
        map_run_length[t] = detector.map_run_length()
        changepoint_prob[t] = detector.changepoint_probability()
        post_break_weight[t] = detector.post_break_weight(break_window)

    return {
        "map_run_length": map_run_length,
        "changepoint_prob": changepoint_prob,
        "post_break_weight": post_break_weight,
        "detector": detector,
    }
