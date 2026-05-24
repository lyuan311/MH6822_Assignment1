"""
analysis/psi_analysis.py
=========================
PSI drift analysis: training vs live distribution.
Applies OCC and MAS thresholds to classify alert level per jurisdiction.
Produces console report + saves Task3_PSIBreakdown.csv.

Run from repo root:
  python analysis/psi_analysis.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from veridian.jurisdiction_config import JurisdictionConfig
from veridian.risk_monitor import compute_psi, assess_drift
from sklearn.metrics import roc_auc_score


DATA_DIR = Path(__file__).parent.parent / "data"


def load_data():
    df_train = pd.read_csv(DATA_DIR / "Task3_TrainingData.csv")
    df_live  = pd.read_csv(DATA_DIR / "Task3_LiveData.csv")
    return df_train, df_live


def psi_report(df_train, df_live, occ_cfg, mas_cfg):
    train_scores = df_train["model_score"].values
    live_scores  = df_live["model_score"].values

    psi_result = compute_psi(train_scores, live_scores, n_buckets=10)
    df_buckets = psi_result.to_dataframe()

    # Save breakdown CSV
    out_path = DATA_DIR / "Task3_PSIBreakdown.csv"
    df_buckets.to_csv(out_path, index=False)
    print(f"[PSI] Breakdown saved → {out_path}")

    # Print bucket table
    print("\n" + "="*75)
    print("  PSI BUCKET BREAKDOWN — Training (2022-2023) vs Live (2025-2026)")
    print("="*75)
    print(f"  {'Bucket':>6}  {'Score Range':<17}  {'Train%':>7}  {'Live%':>7}  "
          f"{'PSI Contrib':>11}  {'Cumulative':>10}")
    print(f"  {'-'*6}  {'-'*17}  {'-'*7}  {'-'*7}  {'-'*11}  {'-'*10}")

    for _, row in df_buckets.iterrows():
        flag = ""
        if row["psi_contribution"] > 0.05:
            flag = " ← HIGH"
        elif row["psi_contribution"] > 0.02:
            flag = " ← moderate"
        print(f"  {int(row['bucket']):>6}  "
              f"{row['score_low']:.3f}–{row['score_high']:.3f}        "
              f"{row['expected_pct']:>7.2%}  {row['actual_pct']:>7.2%}  "
              f"{row['psi_contribution']:>11.4f}  {row['cumulative_psi']:>10.4f}{flag}")

    total = df_buckets["psi_contribution"].sum()
    print(f"\n  Total PSI: {total:.4f}")

    occ_thr = occ_cfg.get_drift_thresholds()
    mas_thr = mas_cfg.get_drift_thresholds()
    occ_status = "RED" if total >= occ_thr.psi_critical else (
                 "YELLOW" if total >= occ_thr.psi_warning else "GREEN")
    mas_status = "RED" if total >= mas_thr.psi_critical else (
                 "YELLOW" if total >= mas_thr.psi_warning else "GREEN")

    print(f"\n  OCC thresholds  warning={occ_thr.psi_warning}  "
          f"critical={occ_thr.psi_critical}  →  {occ_status}")
    print(f"  MAS thresholds  warning={mas_thr.psi_warning}  "
          f"critical={mas_thr.psi_critical}  →  {mas_status}")
    print("="*75)

    # Interpretation
    print("\n  INTERPRETATION")
    print("  " + "-"*50)
    dominant = df_buckets.loc[df_buckets["psi_contribution"].idxmax()]
    print(f"  Dominant bucket: #{int(dominant['bucket'])} "
          f"(score {dominant['score_low']:.3f}–{dominant['score_high']:.3f})")
    print(f"  Training: {dominant['expected_pct']:.1%}  →  "
          f"Live: {dominant['actual_pct']:.1%}  "
          f"(PSI contrib: {dominant['psi_contribution']:.4f})")
    print(f"  Interpretation: {dominant['actual_pct']/dominant['expected_pct']:.1f}x more "
          "applicants fall in the lowest score band in the live period,")
    print("  reflecting the entry of a more credit-stressed population in 2025-2026.")
    print()


def auc_report(df_train, df_live, occ_cfg, mas_cfg):
    auc_train = roc_auc_score(df_train["actual_default"], 1 - df_train["model_score"])
    auc_live  = roc_auc_score(df_live["actual_default"],  1 - df_live["model_score"])
    deg = (auc_train - auc_live) / auc_train * 100

    occ_thr = occ_cfg.get_drift_thresholds()
    mas_thr = mas_cfg.get_drift_thresholds()
    occ_auc = "RED" if deg >= occ_thr.auc_degradation_pct_critical else (
              "YELLOW" if deg >= occ_thr.auc_degradation_pct_warning else "GREEN")
    mas_auc = "RED" if deg >= mas_thr.auc_degradation_pct_critical else (
              "YELLOW" if deg >= mas_thr.auc_degradation_pct_warning else "GREEN")

    print("  AUC PERFORMANCE")
    print("  " + "-"*50)
    print(f"  AUC (training) : {auc_train:.4f}")
    print(f"  AUC (live)     : {auc_live:.4f}")
    print(f"  Degradation    : {deg:.2f}%")
    print(f"  OCC (critical ≥{occ_thr.auc_degradation_pct_critical}%) → {occ_auc}")
    print(f"  MAS (critical ≥{mas_thr.auc_degradation_pct_critical}%) → {mas_auc}")
    print()


def main():
    print("Loading data...")
    df_train, df_live = load_data()

    occ_cfg = JurisdictionConfig.load("OCC")
    mas_cfg = JurisdictionConfig.load("MAS")

    psi_report(df_train, df_live, occ_cfg, mas_cfg)
    auc_report(df_train, df_live, occ_cfg, mas_cfg)


if __name__ == "__main__":
    main()
