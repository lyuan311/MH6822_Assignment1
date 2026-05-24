"""
veridian/jurisdiction_config.py
================================
Loads and validates jurisdiction configuration files.
Each jurisdiction is represented by a YAML file in jurisdiction_config/.

Supported jurisdictions (v1.0):
  - OCC   : OCC Bulletin 2026-13 (US)
  - MAS   : MAS 2024 AI Model Risk Management Guidelines (Singapore)

Usage:
    from veridian.jurisdiction_config import JurisdictionConfig
    cfg = JurisdictionConfig.load("OCC")
    tier = cfg.get_tier_config(1)
    in_scope = cfg.is_model_in_scope("genai")   # False for OCC, True for MAS
"""

from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ── Path resolution ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _REPO_ROOT / "jurisdiction_config"

_JURISDICTION_FILES = {
    "OCC": "OCC_2026_13.yaml",
    "MAS": "MAS_2024_AIRMM.yaml",
}


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class TierConfig:
    definition: str
    validation_frequency_months: int
    requires_independent_validation: bool
    explainability_required: bool
    bias_testing_required: bool
    bias_framework: Optional[str] = None
    explainability_method: List[str] = field(default_factory=list)
    explainability_standard: Optional[str] = None
    ai_ethics_board: bool = False
    adversarial_testing_required: bool = False
    documentation: List[str] = field(default_factory=list)


@dataclass
class SeniorAccountability:
    named_officer_required: bool
    personal_liability: bool
    regime: Optional[str] = None
    default_role: Optional[str] = None
    consequence: Optional[str] = None


@dataclass
class DriftThresholds:
    psi_warning: float
    psi_critical: float
    auc_degradation_pct_warning: float
    auc_degradation_pct_critical: float


@dataclass
class JurisdictionConfig:
    jurisdiction: str
    version: str
    effective_date: str
    scope: Dict[str, Any]
    model_tiers: Dict[str, Any]
    drift_thresholds_raw: Dict[str, Any]
    documentation_requirements: Dict[str, Any]
    senior_accountability_raw: Dict[str, Any]
    regulatory_stability: Dict[str, Any]

    # ── Loader ─────────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, jurisdiction: str) -> "JurisdictionConfig":
        """
        Load a jurisdiction config by short name ('OCC' or 'MAS').
        Falls back to treating jurisdiction as a direct file path if not in registry.
        """
        if jurisdiction.upper() in _JURISDICTION_FILES:
            path = _CONFIG_DIR / _JURISDICTION_FILES[jurisdiction.upper()]
        else:
            path = Path(jurisdiction)

        if not path.exists():
            raise FileNotFoundError(
                f"Jurisdiction config not found: {path}\n"
                f"Available: {list(_JURISDICTION_FILES.keys())}"
            )

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            jurisdiction=raw["jurisdiction"],
            version=raw["version"],
            effective_date=raw["effective_date"],
            scope=raw.get("scope", {}),
            model_tiers=raw.get("model_tiers", {}),
            drift_thresholds_raw=raw.get("drift_thresholds", {}),
            documentation_requirements=raw.get("documentation_requirements", {}),
            senior_accountability_raw=raw.get("senior_accountability", {}),
            regulatory_stability=raw.get("regulatory_stability", {}),
        )

    # ── Accessors ──────────────────────────────────────────────────────────────
    def get_tier_config(self, tier: int) -> TierConfig:
        """Return the TierConfig for the given tier number (1 or 2)."""
        key = f"tier_{tier}"
        if key not in self.model_tiers:
            raise ValueError(
                f"Tier {tier} not defined for jurisdiction {self.jurisdiction}. "
                f"Available: {list(self.model_tiers.keys())}"
            )
        raw = self.model_tiers[key]
        return TierConfig(
            definition=raw.get("definition", ""),
            validation_frequency_months=raw["validation_frequency_months"],
            requires_independent_validation=raw.get("requires_independent_validation", True),
            explainability_required=raw.get("explainability_required", False),
            bias_testing_required=raw.get("bias_testing_required", False),
            bias_framework=raw.get("bias_framework"),
            explainability_method=raw.get("explainability_method", []),
            explainability_standard=raw.get("explainability_standard"),
            ai_ethics_board=raw.get("ai_ethics_board", False),
            adversarial_testing_required=raw.get("adversarial_testing_required", False),
            documentation=raw.get("documentation", []),
        )

    def get_drift_thresholds(self) -> DriftThresholds:
        return DriftThresholds(
            psi_warning=self.drift_thresholds_raw.get("psi_warning", 0.10),
            psi_critical=self.drift_thresholds_raw.get("psi_critical", 0.25),
            auc_degradation_pct_warning=self.drift_thresholds_raw.get("auc_degradation_pct_warning", 5.0),
            auc_degradation_pct_critical=self.drift_thresholds_raw.get("auc_degradation_pct_critical", 10.0),
        )

    def get_senior_accountability(self) -> SeniorAccountability:
        raw = self.senior_accountability_raw
        return SeniorAccountability(
            named_officer_required=raw.get("named_officer_required", False),
            personal_liability=raw.get("personal_liability", False),
            regime=raw.get("regime"),
            default_role=raw.get("default_role"),
            consequence=raw.get("consequence"),
        )

    def is_model_in_scope(self, model_type: str) -> bool:
        """
        Check whether a model type is in scope for this jurisdiction.

        model_type values: 'traditional_ml', 'statistical', 'genai', 'foundation_model'
        """
        type_map = {
            "genai":            "includes_genai",
            "traditional_ml":   "includes_traditional_ml",
            "statistical":      "includes_statistical_models",
            "foundation_model": "includes_foundation_models",
        }
        key = type_map.get(model_type.lower().replace(" ", "_"))
        if key is None:
            # Unknown type — default to in-scope (conservative)
            return True
        return self.scope.get(key, False)

    # ── Utilities ──────────────────────────────────────────────────────────────
    def summary(self) -> str:
        dt = self.get_drift_thresholds()
        sa = self.get_senior_accountability()
        lines = [
            f"Jurisdiction  : {self.jurisdiction}",
            f"Version       : {self.version}  (effective {self.effective_date})",
            f"GenAI in scope: {self.scope.get('includes_genai', False)}",
            f"PSI thresholds: warning={dt.psi_warning}  critical={dt.psi_critical}",
            f"AUC thresholds: warning={dt.auc_degradation_pct_warning}%  critical={dt.auc_degradation_pct_critical}%",
            f"Named officer : {sa.named_officer_required}  (regime: {sa.regime})",
            f"Stability     : {self.regulatory_stability.get('rating', 'UNKNOWN')}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"JurisdictionConfig({self.jurisdiction} v{self.version})"
