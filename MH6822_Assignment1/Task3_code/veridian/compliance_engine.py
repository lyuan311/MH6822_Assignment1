"""
veridian/compliance_engine.py
==============================
Applies jurisdiction-specific compliance rules to model metadata
and drift results, producing a structured ComplianceResult.

Gap severity:
  CRITICAL — regulatory breach; must remediate before next exam
  MAJOR    — significant gap; should remediate within one cycle
  MINOR    — process improvement; document and plan remediation

Overall status:
  COMPLIANT        — no gaps
  GAPS_IDENTIFIED  — major or minor gaps only
  NON_COMPLIANT    — one or more critical gaps
  OUT_OF_SCOPE     — model type not covered by this jurisdiction's rules
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from veridian.jurisdiction_config import JurisdictionConfig
from veridian.risk_monitor import DriftReport


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class ComplianceGap:
    gap_id: str
    category: str           # SCOPE | EXPLAINABILITY | BIAS | DRIFT | VALIDATION | ACCOUNTABILITY | DOCUMENTATION
    description: str
    severity: str           # CRITICAL | MAJOR | MINOR
    remediation: str
    regulatory_reference: str = ""


@dataclass
class ComplianceResult:
    model_id: str
    jurisdiction: str
    jurisdiction_version: str
    overall_status: str      # COMPLIANT | GAPS_IDENTIFIED | NON_COMPLIANT | OUT_OF_SCOPE
    gaps: List[ComplianceGap] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata_snapshot: Dict[str, Any] = field(default_factory=dict)

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def critical_gaps(self) -> List[ComplianceGap]:
        return [g for g in self.gaps if g.severity == "CRITICAL"]

    @property
    def major_gaps(self) -> List[ComplianceGap]:
        return [g for g in self.gaps if g.severity == "MAJOR"]

    @property
    def n_critical(self) -> int:
        return len(self.critical_gaps)

    @property
    def n_major(self) -> int:
        return len(self.major_gaps)

    def summary(self) -> str:
        lines = [
            f"Model          : {self.model_id}",
            f"Jurisdiction   : {self.jurisdiction} ({self.jurisdiction_version})",
            f"Status         : {self.overall_status}",
            f"Critical Gaps  : {self.n_critical}",
            f"Major Gaps     : {self.n_major}",
        ]
        if self.gaps:
            lines.append("\nGaps:")
            for g in self.gaps:
                lines.append(f"  [{g.severity}] {g.gap_id} — {g.description[:80]}...")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "jurisdiction": self.jurisdiction,
            "version": self.jurisdiction_version,
            "status": self.overall_status,
            "n_critical": self.n_critical,
            "n_major": self.n_major,
            "gaps": [
                {
                    "gap_id": g.gap_id,
                    "category": g.category,
                    "severity": g.severity,
                    "description": g.description,
                    "remediation": g.remediation,
                    "reference": g.regulatory_reference,
                }
                for g in self.gaps
            ],
            "warnings": self.warnings,
        }


# ── Compliance engine ──────────────────────────────────────────────────────────
def assess_compliance(
    model_meta: Dict[str, Any],
    drift_report: DriftReport,
    config: JurisdictionConfig,
) -> ComplianceResult:
    """
    Core compliance assessment engine.

    Parameters
    ----------
    model_meta   : dict with model metadata fields (see expected keys below)
    drift_report : DriftReport from risk_monitor.assess_drift()
    config       : JurisdictionConfig for the target jurisdiction

    Expected model_meta keys:
        id                          (str)  model identifier
        name                        (str)  human-readable name
        model_type                  (str)  'traditional_ml' | 'statistical' | 'genai' | 'foundation_model'
        tier                        (int)  1 or 2
        has_explainability_module   (bool) SHAP/LIME/counterfactual output available
        explainability_format       (str)  'feature_importance' | 'counterfactual' | None
        bias_test_completed         (bool)
        bias_framework_used         (str)  e.g. 'ECOA_RegulationB', 'FEAT_Principles'
        months_since_last_validation (int)
        named_smar_officer          (str | None)  name of designated officer
        ai_ethics_board_active      (bool)
        documentation_complete      (list) list of completed doc items

    Returns
    -------
    ComplianceResult
    """
    gaps: List[ComplianceGap] = []
    warnings: List[str] = []
    tier_cfg = config.get_tier_config(model_meta["tier"])
    thresholds = config.get_drift_thresholds()
    sa = config.get_senior_accountability()

    # ── CHECK 0: Scope gate ──────────────────────────────────────────────────
    model_type = model_meta.get("model_type", "traditional_ml")
    if not config.is_model_in_scope(model_type):
        warnings.append(
            f"Model type '{model_type}' is OUT OF SCOPE under "
            f"{config.jurisdiction} {config.version}. "
            f"No compliance assessment will be performed. "
            f"Note: {config.scope.get('note', '')}"
        )
        return ComplianceResult(
            model_id=model_meta["id"],
            jurisdiction=config.jurisdiction,
            jurisdiction_version=config.version,
            overall_status="OUT_OF_SCOPE",
            warnings=warnings,
            metadata_snapshot=model_meta,
        )

    # ── CHECK 1: Explainability (EX) ─────────────────────────────────────────
    if tier_cfg.explainability_required:
        if not model_meta.get("has_explainability_module", False):
            gaps.append(ComplianceGap(
                gap_id="EX-01",
                category="EXPLAINABILITY",
                severity="CRITICAL",
                description=(
                    f"No explainability module detected. "
                    f"{config.jurisdiction} requires "
                    f"{tier_cfg.explainability_method} for Tier {model_meta['tier']} "
                    f"customer-facing models."
                ),
                remediation=(
                    "Integrate SHAP or counterfactual explanation layer. "
                    "Generate sample explanations for 100 adverse decisions "
                    "and verify they meet the customer-intelligible standard."
                ),
                regulatory_reference=f"{config.jurisdiction} {config.version} — Explainability Principle",
            ))
        elif model_meta.get("explainability_format") == "feature_importance":
            # Feature importance (e.g. raw SHAP values) ≠ customer-intelligible under MAS
            if tier_cfg.explainability_standard == "customer_intelligible":
                gaps.append(ComplianceGap(
                    gap_id="EX-02",
                    category="EXPLAINABILITY",
                    severity="MAJOR",
                    description=(
                        "Explainability module exists but produces feature-importance output "
                        "(e.g. raw SHAP values). MAS requires customer-intelligible explanations — "
                        "plain-language counterfactuals, not numerical feature weights."
                    ),
                    remediation=(
                        "Augment SHAP output with a counterfactual explanation layer. "
                        "Test explanations with a sample of customers for intelligibility."
                    ),
                    regulatory_reference="MAS 2024 AI MRM — Explainability; FEAT Transparency Principle",
                ))

    # ── CHECK 2: Bias / Fairness testing (BT) ───────────────────────────────
    if tier_cfg.bias_testing_required:
        if not model_meta.get("bias_test_completed", False):
            gaps.append(ComplianceGap(
                gap_id="BT-01",
                category="BIAS",
                severity="CRITICAL",
                description=(
                    f"Bias testing not completed under {tier_cfg.bias_framework}. "
                    f"Required for Tier {model_meta['tier']} models."
                ),
                remediation=(
                    f"Conduct full {tier_cfg.bias_framework} fairness analysis. "
                    "Document results, including approval rate disparity "
                    "by all relevant demographic groups. Retain records."
                ),
                regulatory_reference=f"{config.jurisdiction} {config.version} — {tier_cfg.bias_framework}",
            ))
        else:
            # Check framework alignment
            used = model_meta.get("bias_framework_used", "")
            required = tier_cfg.bias_framework or ""
            if used and required and used.lower() != required.lower():
                gaps.append(ComplianceGap(
                    gap_id="BT-02",
                    category="BIAS",
                    severity="MAJOR",
                    description=(
                        f"Bias testing was completed under '{used}' framework, "
                        f"but {config.jurisdiction} requires '{required}'. "
                        "Frameworks differ in scope (e.g. FEAT covers proxy discrimination; "
                        "ECOA does not)."
                    ),
                    remediation=(
                        f"Re-run bias analysis under {required} framework. "
                        "Pay particular attention to proxy variables (e.g. postal code) "
                        "if switching to FEAT Principles."
                    ),
                    regulatory_reference=f"{config.jurisdiction} {config.version} — Fairness",
                ))

    # ── CHECK 3: Data drift / PSI (DR) ──────────────────────────────────────
    if drift_report.overall_alert == "RED":
        gaps.append(ComplianceGap(
            gap_id="DR-01",
            category="DRIFT",
            severity="CRITICAL",
            description=(
                f"Model PSI = {drift_report.psi:.4f}, exceeding the "
                f"{config.jurisdiction} critical threshold "
                f"({thresholds.psi_critical}). "
                f"AUC degradation = {drift_report.auc_degradation_pct:.1f}%. "
                "Significant distribution shift detected between training "
                "and live populations."
            ),
            remediation=(
                "Suspend model from production pending full revalidation. "
                "Investigate root cause of distribution shift. "
                "Notify CMRO and (under MAS) named SMAR officer immediately."
            ),
            regulatory_reference=f"{config.jurisdiction} {config.version} — Ongoing Monitoring",
        ))
    elif drift_report.overall_alert == "YELLOW":
        warnings.append(
            f"PSI = {drift_report.psi:.4f} exceeds {config.jurisdiction} warning "
            f"threshold ({thresholds.psi_warning}). Monitor closely. "
            f"AUC degradation: {drift_report.auc_degradation_pct:.1f}%. "
            "Consider scheduling early revalidation."
        )

    # ── CHECK 4: Validation currency (VL) ───────────────────────────────────
    months_since = model_meta.get("months_since_last_validation", 0)
    required_freq = tier_cfg.validation_frequency_months
    if months_since > required_freq:
        overdue = months_since - required_freq
        gaps.append(ComplianceGap(
            gap_id="VL-01",
            category="VALIDATION",
            severity="MAJOR",
            description=(
                f"Last validation was {months_since} months ago. "
                f"{config.jurisdiction} requires Tier {model_meta['tier']} "
                f"models to be validated every {required_freq} months. "
                f"Model is {overdue} month(s) overdue."
            ),
            remediation=(
                f"Schedule model validation immediately. "
                f"Target completion within 30 days. "
                f"Document the delay and its business justification."
            ),
            regulatory_reference=f"{config.jurisdiction} {config.version} — Validation Frequency",
        ))

    # ── CHECK 5: Senior accountability / SMAR (SA) ──────────────────────────
    if sa.named_officer_required:
        if not model_meta.get("named_smar_officer"):
            gaps.append(ComplianceGap(
                gap_id="SA-01",
                category="ACCOUNTABILITY",
                severity="CRITICAL",
                description=(
                    f"No named {sa.regime} officer assigned to this model. "
                    f"{config.jurisdiction} requires a designated "
                    f"{sa.default_role} to be personally accountable "
                    "for AI model governance."
                ),
                remediation=(
                    f"Assign a named {sa.default_role} as the accountable "
                    f"officer under {sa.regime}. Record the assignment in the "
                    "model registry. Consequence of non-compliance: "
                    f"{sa.consequence}."
                ),
                regulatory_reference=f"MAS SMAR; MAS 2024 AI MRM — Accountability Principle",
            ))

    # ── CHECK 6: AI Ethics Board (AE) ───────────────────────────────────────
    if tier_cfg.ai_ethics_board:
        if not model_meta.get("ai_ethics_board_active", False):
            gaps.append(ComplianceGap(
                gap_id="AE-01",
                category="DOCUMENTATION",
                severity="MAJOR",
                description=(
                    f"No AI Ethics Board or equivalent oversight body active. "
                    f"{config.jurisdiction} requires an ethics board "
                    f"for Tier {model_meta['tier']} AI models."
                ),
                remediation=(
                    "Establish an AI Ethics Board or extend an existing "
                    "Risk Committee's mandate to cover AI ethics. "
                    "Maintain board minutes for each model reviewed."
                ),
                regulatory_reference="MAS 2024 AI MRM — Governance Structure",
            ))

    # ── CHECK 7: Adversarial testing for GenAI (AT) ─────────────────────────
    if tier_cfg.adversarial_testing_required and model_type in ("genai", "foundation_model"):
        if not model_meta.get("adversarial_testing_completed", False):
            gaps.append(ComplianceGap(
                gap_id="AT-01",
                category="VALIDATION",
                severity="CRITICAL",
                description=(
                    "GenAI / foundation model has not undergone adversarial testing. "
                    "MAS requires prompt injection testing, hallucination rate analysis, "
                    "and jailbreak resistance evaluation for customer-facing GenAI."
                ),
                remediation=(
                    "Conduct adversarial testing suite: prompt injection, "
                    "hallucination benchmarking (minimum 500 test cases), "
                    "and output consistency testing. Document results."
                ),
                regulatory_reference="MAS 2024 AI MRM — Robustness; GenAI Annex",
            ))

    # ── Determine overall status ─────────────────────────────────────────────
    if any(g.severity == "CRITICAL" for g in gaps):
        status = "NON_COMPLIANT"
    elif gaps:
        status = "GAPS_IDENTIFIED"
    else:
        status = "COMPLIANT"

    return ComplianceResult(
        model_id=model_meta["id"],
        jurisdiction=config.jurisdiction,
        jurisdiction_version=config.version,
        overall_status=status,
        gaps=gaps,
        warnings=warnings,
        metadata_snapshot=model_meta,
    )
