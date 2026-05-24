"""
demo/run_demo.py
=================
End-to-end demonstration of VeridianMRM jurisdiction-aware compliance.

Scenario: SCB-CS-001 — Standard Chartered Bank Retail Credit Scoring Model v2.3
  - Deployed in both New York Branch (OCC) and Singapore (MAS)
  - Same model metadata, same data — different compliance outcomes

Run from repo root:
  python demo/run_demo.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Add repo root to path so veridian package is importable ──────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from veridian.jurisdiction_config import JurisdictionConfig
from veridian.risk_monitor import assess_drift
from veridian.compliance_engine import assess_compliance


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODEL METADATA
#    Represents what a bank's model registry would contain for SCB-CS-001
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_META = {
    "id":                          "SCB-CS-001",
    "name":                        "Retail Credit Limit Scoring Model v2.3",
    "model_type":                  "traditional_ml",     # gradient-boosted tree
    "tier":                        1,                    # Tier 1: consumer credit
    # Explainability
    "has_explainability_module":   False,   # No SHAP/LIME deployed
    "explainability_format":       None,
    # Bias testing
    "bias_test_completed":         True,
    "bias_framework_used":         "ECOA_RegulationB",   # US framework used; not FEAT
    # Validation currency
    "months_since_last_validation":14,      # Overdue under 6-month requirement
    # Senior accountability
    "named_smar_officer":          None,    # Not yet assigned
    # Governance
    "ai_ethics_board_active":      False,
    "adversarial_testing_completed": False,
    "documentation_complete":      ["model_inventory_entry", "validation_report"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD DATA  (uses pre-generated CSVs if available, else generates synthetic)
# ═══════════════════════════════════════════════════════════════════════════════
def load_scores():
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "Task3_TrainingData.csv"
    live_path  = data_dir / "Task3_LiveData.csv"

    if train_path.exists() and live_path.exists():
        df_train = pd.read_csv(train_path)
        df_live  = pd.read_csv(live_path)
        print(f"[DATA] Loaded from CSV  "
              f"(train: {len(df_train):,} rows, live: {len(df_live):,} rows)")
        train_scores = df_train["model_score"].values
        live_scores  = df_live["model_score"].values
        # Compute AUC from data
        from sklearn.metrics import roc_auc_score
        auc_train = roc_auc_score(df_train["actual_default"], 1 - df_train["model_score"])
        auc_live  = roc_auc_score(df_live["actual_default"],  1 - df_live["model_score"])
    else:
        print("[DATA] CSV files not found — using compact synthetic scores")
        rng = np.random.default_rng(42)
        train_scores = np.clip(rng.beta(2, 5, 5000) + 0.35, 0, 1)
        rng2 = np.random.default_rng(99)
        live_scores  = np.clip(rng2.beta(1.5, 4, 3000) + 0.20, 0, 1)
        auc_train, auc_live = 0.780, 0.741

    return train_scores, live_scores, auc_train, auc_live


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
COLOURS = {
    "GREEN":          "\033[92m",
    "YELLOW":         "\033[93m",
    "RED":            "\033[91m",
    "COMPLIANT":      "\033[92m",
    "GAPS_IDENTIFIED":"\033[93m",
    "NON_COMPLIANT":  "\033[91m",
    "OUT_OF_SCOPE":   "\033[94m",
    "CRITICAL":       "\033[91m",
    "MAJOR":          "\033[93m",
    "MINOR":          "\033[94m",
    "RESET":          "\033[0m",
}

def colour(text: str, key: str) -> str:
    return f"{COLOURS.get(key, '')}{text}{COLOURS['RESET']}"

def banner(title: str, width: int = 60):
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)

def section(title: str):
    print(f"\n── {title} {'─' * (50 - len(title))}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN DEMO
# ═══════════════════════════════════════════════════════════════════════════════
def run_demo():
    banner("VeridianMRM — Jurisdiction-Aware Compliance Demo")
    print(f"  Model : {MODEL_META['id']} — {MODEL_META['name']}")
    print(f"  Type  : {MODEL_META['model_type']}  |  Tier: {MODEL_META['tier']}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_scores, live_scores, auc_train, auc_live = load_scores()

    # ── Load jurisdiction configs ─────────────────────────────────────────────
    occ_config = JurisdictionConfig.load("OCC")
    mas_config = JurisdictionConfig.load("MAS")

    print(f"\n  Jurisdictions loaded:")
    print(f"    {occ_config}")
    print(f"    {mas_config}")

    # ── Drift assessment ──────────────────────────────────────────────────────
    banner("DRIFT ASSESSMENT")
    occ_drift = assess_drift("SCB-CS-001", train_scores, live_scores,
                              auc_train, auc_live, occ_config)
    mas_drift = assess_drift("SCB-CS-001", train_scores, live_scores,
                              auc_train, auc_live, mas_config)

    section("Population Stability Index")
    print(f"  Total PSI : {occ_drift.psi:.4f}")
    print(f"  OCC  →  warning ≥ {occ_config.get_drift_thresholds().psi_warning}  "
          f"critical ≥ {occ_config.get_drift_thresholds().psi_critical}  "
          f"→  {colour(occ_drift.psi_alert, occ_drift.psi_alert)}")
    print(f"  MAS  →  warning ≥ {mas_config.get_drift_thresholds().psi_warning}  "
          f"critical ≥ {mas_config.get_drift_thresholds().psi_critical}  "
          f"→  {colour(mas_drift.psi_alert, mas_drift.psi_alert)}")

    section("AUC Performance")
    print(f"  AUC (training): {auc_train:.4f}")
    print(f"  AUC (live)    : {auc_live:.4f}")
    print(f"  Degradation   : {occ_drift.auc_degradation_pct:.2f}%")
    print(f"  OCC  →  critical ≥ {occ_config.get_drift_thresholds().auc_degradation_pct_critical}%  "
          f"→  {colour(occ_drift.auc_alert, occ_drift.auc_alert)}")
    print(f"  MAS  →  critical ≥ {mas_config.get_drift_thresholds().auc_degradation_pct_critical}%  "
          f"→  {colour(mas_drift.auc_alert, mas_drift.auc_alert)}")

    # ── Compliance assessment ─────────────────────────────────────────────────
    banner("COMPLIANCE ASSESSMENT")
    occ_result = assess_compliance(MODEL_META, occ_drift, occ_config)
    mas_result = assess_compliance(MODEL_META, mas_drift, mas_config)

    for result in [occ_result, mas_result]:
        section(f"{result.jurisdiction}  ({result.jurisdiction_version})")
        status_str = colour(result.overall_status, result.overall_status)
        print(f"  Overall Status : {status_str}")
        print(f"  Critical Gaps  : {result.n_critical}")
        print(f"  Major Gaps     : {result.n_major}")

        if result.gaps:
            print()
            for gap in result.gaps:
                sev = colour(f"[{gap.severity}]", gap.severity)
                print(f"  {sev} {gap.gap_id}  {gap.category}")
                print(f"       {gap.description[:90]}")
                print(f"       → {gap.remediation[:80]}")
                print()

        if result.warnings:
            for w in result.warnings:
                print(f"  {colour('⚠  WARNING', 'YELLOW')}  {w[:90]}")

    # ── Side-by-side comparison ───────────────────────────────────────────────
    banner("JURISDICTION COMPARISON SUMMARY")
    print(f"\n  {'Check':<30}  {'OCC 2026-13':<22}  {'MAS 2024 AI MRM'}")
    print(f"  {'-'*30}  {'-'*22}  {'-'*22}")

    checks = [
        ("GenAI in scope?",        "No (excluded)",        "Yes (Tier 1)"),
        ("Explainability required?","No",                   "Yes (counterfactual)"),
        (f"PSI = {occ_drift.psi:.3f}",
                                   colour(occ_drift.psi_alert, occ_drift.psi_alert),
                                   colour(mas_drift.psi_alert, mas_drift.psi_alert)),
        (f"AUC deg. {occ_drift.auc_degradation_pct:.1f}%",
                                   colour(occ_drift.auc_alert, occ_drift.auc_alert),
                                   colour(mas_drift.auc_alert, mas_drift.auc_alert)),
        ("Validation overdue?",     "Yes (14 vs 6 mo)",    "Yes (14 vs 6 mo)"),
        ("Named SMAR officer?",     "N/A (not required)",  colour("MISSING", "RED")),
        ("Overall Status",
                                   colour(occ_result.overall_status, occ_result.overall_status),
                                   colour(mas_result.overall_status, mas_result.overall_status)),
    ]
    for label, occ_val, mas_val in checks:
        print(f"  {label:<30}  {occ_val:<35}  {mas_val}")

    # ── Key finding ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  KEY FINDING")
    print("═" * 60)
    print()
    print("  Same model. Same data. Different compliance outcomes.")
    print()
    occ_gaps_n = occ_result.n_critical + occ_result.n_major
    mas_gaps_n = mas_result.n_critical + mas_result.n_major
    print(f"  OCC : {occ_result.overall_status}  ({occ_gaps_n} gaps)")
    print(f"  MAS : {mas_result.overall_status}  ({mas_gaps_n} gaps)")
    print()
    print("  The 3 additional MAS gaps (EX-01, SA-01, BT-02) are not")
    print("  model quality failures — they are jurisdiction-specific")
    print("  governance requirements that OCC does not mandate.")
    print("  A bank monitoring only against OCC standards would present")
    print("  a clean dashboard while non-compliant under MAS rules.")
    print()
    print("  This is the problem VeridianMRM is designed to solve.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
