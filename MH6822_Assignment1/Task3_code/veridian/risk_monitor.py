"""
veridian/risk_monitor.py
=========================
Computes Population Stability Index (PSI), AUC degradation,
and drift alert levels against jurisdiction-specific thresholds.

PSI interpretation (industry standard):
  < 0.10  : No significant change    (GREEN)
  0.10 – 0.20 : Some change, monitor  (YELLOW)
  > 0.20  : Significant change       (RED)

Note: MAS and OCC use different warning/critical split points.
Both are applied in assess_drift() based on the jurisdiction config.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from veridian.jurisdiction_config import JurisdictionConfig, DriftThresholds


# ── Result data classes ────────────────────────────────────────────────────────
@dataclass
class PSIBucket:
    bucket_id: int
    score_range: Tuple[float, float]
    expected_pct: float      # training distribution
    actual_pct: float        # live distribution
    psi_contribution: float


@dataclass
class PSIResult:
    total_psi: float
    buckets: List[PSIBucket]
    n_training: int
    n_live: int

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "bucket": b.bucket_id,
                "score_low": round(b.score_range[0], 4),
                "score_high": round(b.score_range[1], 4),
                "expected_pct": round(b.expected_pct, 4),
                "actual_pct": round(b.actual_pct, 4),
                "psi_contribution": round(b.psi_contribution, 4),
                "cumulative_psi": round(
                    sum(x.psi_contribution for x in self.buckets[:i+1]), 4
                ),
            }
            for i, b in enumerate(self.buckets)
        ])

    def alert_level(self, thresholds: DriftThresholds) -> str:
        if self.total_psi >= thresholds.psi_critical:
            return "RED"
        elif self.total_psi >= thresholds.psi_warning:
            return "YELLOW"
        return "GREEN"


@dataclass
class DriftReport:
    model_id: str
    jurisdiction: str
    psi_result: PSIResult
    auc_train: float
    auc_live: float
    auc_degradation_pct: float
    psi_alert: str        # GREEN / YELLOW / RED
    auc_alert: str        # GREEN / YELLOW / RED
    overall_alert: str    # most severe of psi and auc
    thresholds: DriftThresholds

    @property
    def psi(self) -> float:
        return self.psi_result.total_psi

    def summary(self) -> str:
        lines = [
            f"Model          : {self.model_id}",
            f"Jurisdiction   : {self.jurisdiction}",
            f"PSI            : {self.psi:.4f}  →  {self.psi_alert}",
            f"  Warning  ≥ {self.thresholds.psi_warning}  |  Critical ≥ {self.thresholds.psi_critical}",
            f"AUC Train      : {self.auc_train:.4f}",
            f"AUC Live       : {self.auc_live:.4f}",
            f"AUC Degradation: {self.auc_degradation_pct:.2f}%  →  {self.auc_alert}",
            f"  Warning  ≥ {self.thresholds.auc_degradation_pct_warning}%"
            f"  |  Critical ≥ {self.thresholds.auc_degradation_pct_critical}%",
            f"Overall Alert  : {self.overall_alert}",
        ]
        return "\n".join(lines)


# ── Core functions ─────────────────────────────────────────────────────────────
def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_buckets: int = 10,
) -> PSIResult:
    """
    Compute Population Stability Index.

    Parameters
    ----------
    expected : array-like
        Model scores from the training / reference period.
    actual : array-like
        Model scores from the live / monitoring period.
    n_buckets : int
        Number of equal-frequency buckets (default 10).

    Returns
    -------
    PSIResult containing total PSI and per-bucket breakdown.
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)

    # Build bucket breakpoints on the expected (training) distribution
    percentiles = np.linspace(0, 100, n_buckets + 1)
    breakpoints = np.percentile(expected, percentiles)
    # Extend edges to capture full range
    breakpoints[0]  -= 1e-6
    breakpoints[-1] += 1e-6

    # Count observations per bucket
    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]

    n_exp = len(expected)
    n_act = len(actual)

    # Convert to proportions; add small epsilon to avoid log(0)
    exp_pct = exp_counts / n_exp + 1e-8
    act_pct = act_counts / n_act + 1e-8

    psi_contributions = (act_pct - exp_pct) * np.log(act_pct / exp_pct)

    buckets = [
        PSIBucket(
            bucket_id=i + 1,
            score_range=(breakpoints[i], breakpoints[i + 1]),
            expected_pct=float(exp_pct[i]),
            actual_pct=float(act_pct[i]),
            psi_contribution=float(psi_contributions[i]),
        )
        for i in range(n_buckets)
    ]

    return PSIResult(
        total_psi=float(psi_contributions.sum()),
        buckets=buckets,
        n_training=n_exp,
        n_live=n_act,
    )


def compute_auc_degradation(auc_train: float, auc_live: float) -> float:
    """Percentage degradation of AUC from training to live."""
    if auc_train == 0:
        return 0.0
    return (auc_train - auc_live) / auc_train * 100


def _alert_level(value: float, warning: float, critical: float) -> str:
    if value >= critical:
        return "RED"
    elif value >= warning:
        return "YELLOW"
    return "GREEN"


def _most_severe(levels: List[str]) -> str:
    priority = {"RED": 3, "YELLOW": 2, "GREEN": 1}
    return max(levels, key=lambda l: priority.get(l, 0))


def assess_drift(
    model_id: str,
    train_scores: np.ndarray,
    live_scores: np.ndarray,
    auc_train: float,
    auc_live: float,
    config: JurisdictionConfig,
    n_buckets: int = 10,
) -> DriftReport:
    """
    Compute PSI, AUC degradation, and generate a jurisdiction-specific DriftReport.

    Parameters
    ----------
    model_id    : identifier for the model being assessed
    train_scores: model output scores from training period
    live_scores : model output scores from live period
    auc_train   : AUC computed on training period (vs actual defaults)
    auc_live    : AUC computed on live period (vs actual defaults)
    config      : JurisdictionConfig (OCC or MAS)
    n_buckets   : PSI bucket count (default 10)

    Returns
    -------
    DriftReport with alert levels calibrated to the jurisdiction config.
    """
    psi_result = compute_psi(train_scores, live_scores, n_buckets)
    auc_deg    = compute_auc_degradation(auc_train, auc_live)
    thresholds = config.get_drift_thresholds()

    psi_alert = psi_result.alert_level(thresholds)
    auc_alert = _alert_level(
        auc_deg,
        thresholds.auc_degradation_pct_warning,
        thresholds.auc_degradation_pct_critical,
    )

    return DriftReport(
        model_id=model_id,
        jurisdiction=config.jurisdiction,
        psi_result=psi_result,
        auc_train=auc_train,
        auc_live=auc_live,
        auc_degradation_pct=auc_deg,
        psi_alert=psi_alert,
        auc_alert=auc_alert,
        overall_alert=_most_severe([psi_alert, auc_alert]),
        thresholds=thresholds,
    )


# ── Time-series monitor ────────────────────────────────────────────────────────
def classify_monthly_series(
    df_metrics: pd.DataFrame,
    config: JurisdictionConfig,
) -> pd.DataFrame:
    """
    Apply jurisdiction thresholds to a monthly performance dataframe.

    Expected columns: month, auc_roc, psi

    Returns the dataframe with added columns:
        psi_alert, auc_alert, overall_alert
    """
    thresholds = config.get_drift_thresholds()
    df = df_metrics.copy()

    df["psi_alert"] = df["psi"].apply(
        lambda v: _alert_level(v, thresholds.psi_warning, thresholds.psi_critical)
    )

    # AUC degradation relative to first observation (baseline)
    baseline_auc = df["auc_roc"].iloc[0]
    df["auc_degradation_pct"] = (baseline_auc - df["auc_roc"]) / baseline_auc * 100
    df["auc_alert"] = df["auc_degradation_pct"].apply(
        lambda v: _alert_level(v, thresholds.auc_degradation_pct_warning,
                               thresholds.auc_degradation_pct_critical)
    )

    df["overall_alert"] = df.apply(
        lambda r: _most_severe([r["psi_alert"], r["auc_alert"]]), axis=1
    )
    df["jurisdiction"] = config.jurisdiction

    return df
