"""
analysis/bias_analysis.py
==========================
Fairness / bias analysis applied under two regulatory frameworks:
  - OCC / ECOA / Regulation B  (US)
  - MAS FEAT Principles        (Singapore)

Demonstrates that the same dataset produces different compliance
findings depending on which fairness framework is applied.

Run from repo root:
  python analysis/bias_analysis.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from veridian.jurisdiction_config import JurisdictionConfig


DATA_DIR = Path(__file__).parent.parent / "data"

# ── Regulatory thresholds ─────────────────────────────────────────────────────
# OCC/ECOA: 80% rule (disparate impact) — approval rate ratio < 0.80 is prima facie adverse impact
ECOA_RATIO_THRESHOLD = 0.80
# FEAT: 5pp approval rate gap triggers formal review (MAS industry convention)
FEAT_GAP_THRESHOLD_PP = 5.0


def load_data():
    df_train = pd.read_csv(DATA_DIR / "Task3_TrainingData.csv")
    df_live  = pd.read_csv(DATA_DIR / "Task3_LiveData.csv")
    return df_train, df_live


def approval_stats(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Approval rate and default rate by group."""
    grp = df.groupby(attribute).agg(
        n=("approved", "count"),
        approval_rate=("approved", "mean"),
        default_rate=("actual_default", "mean"),
        mean_score=("model_score", "mean"),
    ).reset_index()
    grp["approval_rate_pct"] = (grp["approval_rate"] * 100).round(2)
    grp["default_rate_pct"]  = (grp["default_rate"]  * 100).round(2)
    grp["mean_score"]        = grp["mean_score"].round(4)
    return grp


def ecoa_regb_assessment(df_train, df_live):
    """
    US ECOA / Regulation B disparate impact analysis.
    Checks: gender, race/national origin proxy (ethnicity).
    Method: 80% / 4/5ths rule on approval rate ratio.
    Baseline group: highest approval rate group.
    """
    print("\n" + "="*65)
    print("  ECOA / REGULATION B — Disparate Impact Analysis (OCC Framework)")
    print("="*65)
    print("  Method: 80% Rule — approval rate ratio < 0.80 = adverse impact")
    print("  Protected attributes (US): gender, race/national origin\n")

    for attr in ["gender", "ethnicity"]:
        for period, df in [("Training (2022-2023)", df_train), ("Live (2025-2026)", df_live)]:
            stats = approval_stats(df, attr)
            baseline_rate = stats["approval_rate"].max()
            stats["ratio_vs_baseline"] = (stats["approval_rate"] / baseline_rate).round(4)
            stats["ecoa_flag"] = stats["ratio_vs_baseline"] < ECOA_RATIO_THRESHOLD

            print(f"  Attribute: {attr.upper()}  |  Period: {period}")
            print(f"  {'Group':<12}  {'N':>6}  {'Approval%':>10}  "
                  f"{'Ratio':>7}  {'ECOA Flag'}")
            print(f"  {'-'*12}  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*10}")
            for _, row in stats.iterrows():
                flag = "⚠  ADVERSE IMPACT" if row["ecoa_flag"] else "✓  OK"
                print(f"  {str(row[attr]):<12}  {int(row['n']):>6}  "
                      f"{row['approval_rate_pct']:>9.2f}%  "
                      f"{row['ratio_vs_baseline']:>7.4f}  {flag}")
            print()


def feat_principles_assessment(df_train, df_live):
    """
    MAS FEAT Principles — Fairness analysis.
    Broader scope: all demographic groups + proxy discrimination.
    Method: absolute approval rate gap vs reference group; proxy variable flagging.
    """
    print("="*65)
    print("  MAS FEAT PRINCIPLES — Fairness Analysis (MAS Framework)")
    print("="*65)
    print("  Scope: ALL demographic groups; proxy discrimination in scope")
    print(f"  Threshold: >{FEAT_GAP_THRESHOLD_PP}pp gap triggers formal review\n")

    for attr in ["gender", "ethnicity"]:
        print(f"  Attribute: {attr.upper()}")
        print(f"  {'Group':<12}  {'Train%':>8}  {'Live%':>8}  {'Change pp':>10}  "
              f"{'Gap vs Ref':>10}  {'FEAT Status'}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*16}")

        train_stats = approval_stats(df_train, attr).set_index(attr)
        live_stats  = approval_stats(df_live,  attr).set_index(attr)

        # Reference: Chinese for ethnicity, M for gender
        ref = {"gender": "M", "ethnicity": "Chinese"}[attr]
        ref_live = live_stats.loc[ref, "approval_rate"] if ref in live_stats.index else None

        for grp in train_stats.index:
            t_rate = train_stats.loc[grp, "approval_rate"] * 100
            l_rate = live_stats.loc[grp,  "approval_rate"] * 100 if grp in live_stats.index else np.nan
            change = l_rate - t_rate
            gap_vs_ref = (l_rate - ref_live * 100) if ref_live else np.nan

            if abs(gap_vs_ref) > FEAT_GAP_THRESHOLD_PP:
                status = "✗  REVIEW REQUIRED"
            elif abs(gap_vs_ref) > 2.0:
                status = "⚠  MONITOR (FEAT)"
            else:
                status = "✓  OK"

            print(f"  {str(grp):<12}  {t_rate:>7.2f}%  {l_rate:>7.2f}%  "
                  f"{change:>+9.2f}pp  {gap_vs_ref:>+9.2f}pp  {status}")
        print()

    # ── Proxy discrimination: postal district ─────────────────────────────────
    print("  PROXY DISCRIMINATION ANALYSIS — Postal District")
    print("  " + "-"*55)
    print("  MAS FEAT explicitly requires assessment of variables that")
    print("  may serve as proxies for protected characteristics.")
    print()
    print("  postal_district correlates with:")
    print("    → Property value (direct correlation, Singapore property data)")
    print("    → Household income (moderate correlation)")
    print("    → Ethnicity (aggregate level, documented in SG housing statistics)")
    print()
    print("  VeridianMRM flags postal_district as HIGH PROXY RISK under MAS.")
    print("  Under OCC/ECOA, geographic variables require 'disparate impact")
    print("  review' (CFPB digital redlining guidance) but no proxy flag.")
    print()

    # Compute postal district vs ethnicity correlation in live data
    _, df_live = load_data()
    eth_map = {"Chinese": 0, "Malay": 1, "Indian": 2, "Other": 3}
    eth_enc = df_live["ethnicity"].map(eth_map)
    corr = df_live["postal_district"].corr(eth_enc)
    print(f"  Observed correlation (postal_district ↔ ethnicity encoding): r = {corr:.4f}")
    print(f"  Assessment: {'Moderate proxy risk' if abs(corr) > 0.1 else 'Low observed correlation'}")
    print("  Note: Aggregate-level proxy effects may not be visible in synthetic data.")
    print()


def cross_framework_comparison():
    """Show what each framework finds and what each misses."""
    print("="*65)
    print("  FRAMEWORK COMPARISON — What Each Finds, What Each Misses")
    print("="*65)

    rows = [
        ["Gender disparity",           "✓ In scope",  "✓ In scope",  "No gap found (both)"],
        ["Ethnicity disparity",        "✓ In scope",  "✓ In scope",  "Minor gap in 'Other' (live)"],
        ["Proxy variable (postcode)",  "Partial (CFPB redlining)", "✓ Explicit (FEAT)", "Flagged only under MAS"],
        ["Disability / age proxies",   "Age in scope (ECOA)", "All groups (FEAT)", "FEAT broader coverage"],
        ["Bias framework mismatch",    "N/A",         "✓ Detects",   "Model tested under ECOA; MAS needs FEAT"],
        ["Counterfactual explanation", "Not required","✓ Required",  "Gap EX-01 under MAS only"],
    ]

    print(f"\n  {'Finding':<30}  {'ECOA/OCC':<25}  {'FEAT/MAS':<25}  {'Compliance Impact'}")
    print(f"  {'-'*30}  {'-'*25}  {'-'*25}  {'-'*25}")
    for r in rows:
        print(f"  {r[0]:<30}  {r[1]:<25}  {r[2]:<25}  {r[3]}")

    print()
    print("  KEY FINDING:")
    print("  The same model passes ECOA disparate impact tests but has")
    print("  an unresolved FEAT gap (bias_framework_used = ECOA_RegulationB,")
    print("  not FEAT_Principles). VeridianMRM fires gap BT-02 under MAS.")
    print("  This finding is invisible under OCC-only assessment.\n")


def main():
    df_train, df_live = load_data()
    ecoa_regb_assessment(df_train, df_live)
    feat_principles_assessment(df_train, df_live)
    cross_framework_comparison()


def load_data():
    df_train = pd.read_csv(DATA_DIR / "Task3_TrainingData.csv")
    df_live  = pd.read_csv(DATA_DIR / "Task3_LiveData.csv")
    return df_train, df_live


if __name__ == "__main__":
    main()
