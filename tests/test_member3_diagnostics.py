"""
Automated Unit and Integration Tests for EV Charging Diagnostics:
Safety, Status Labels, Evidence Breakdown, SOPs, Q&A, and PDF Generation.
Validated across all 8 required evaluation scenarios.
"""

import sys
import unittest
from pathlib import Path

# Resolve project paths
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from src.diagnostics.schema import (
    VisualInspectionStatus,
    OperationalStatus,
    SeverityLevel,
    MaintenancePriority,
    ChargerContext,
    CVInferenceResult,
)
from src.diagnostics.status_assessor import StatusAssessor
from src.diagnostics.engine import EVDiagnosticEngine
from src.diagnostics.report_generator import ReportGenerator
from src.diagnostics import diagnose_image, diagnose_cv_result


class TestMember3Diagnostics(unittest.TestCase):

    def setUp(self):
        self.engine = EVDiagnosticEngine()

    # -----------------------------------------------------------------
    # Scenario 1: Healthy charger correctly classified
    # -----------------------------------------------------------------
    def test_scenario1_healthy_charger_correctly_classified(self):
        """Scenario 1: Healthy charger with high confidence shows preferred labels and disclaimer."""
        cv_res = {
            "status": "success",
            "condition": "healthy",
            "condition_confidence": 0.99,
            "component": "charger",
            "component_confidence": 0.96,
            "is_fault_detected": False
        }
        res = self.engine.diagnose_from_cv(cv_res, station_id="EVCS-TEST-CHARGER-01")

        self.assertEqual(res.visual_inspection_status, VisualInspectionStatus.NO_DEFECT_DETECTED)
        self.assertEqual(res.visual_inspection_status.value, "NO VISIBLE DEFECT DETECTED")
        self.assertEqual(res.overall_severity, SeverityLevel.NONE_DETECTED)
        self.assertEqual(res.overall_severity.value, "NO VISIBLE DAMAGE IDENTIFIED")
        self.assertEqual(res.maintenance_priority, MaintenancePriority.ROUTINE)
        self.assertEqual(res.maintenance_priority.value, "ROUTINE MAINTENANCE")
        self.assertEqual(res.status_display_label, "No Visible Defect Detected")
        self.assertIsNone(res.dispatch_sla_hours)
        self.assertEqual(res.dispatch_sla_display, "Not assigned")
        self.assertIn("cannot be determined from optical imagery", res.safety_hazards[1].lower())
        self.assertTrue(len(res.disclaimer) > 20)

    # -----------------------------------------------------------------
    # Scenario 2: Broken cord with exposed wires
    # -----------------------------------------------------------------
    def test_scenario2_broken_cord_exposed_wires(self):
        """Scenario 2: Broken cord triggers CRITICAL severity, LOTO, and evidence breakdown."""
        cv_res = {
            "status": "success",
            "condition": "broken",
            "condition_confidence": 0.98,
            "component": "cord",
            "component_confidence": 0.95,
            "is_fault_detected": True,
            "diagnostic_hint": "Severe outer jacket tear exposing internal cable layers."
        }
        res = self.engine.diagnose_from_cv(cv_res, station_id="EVCS-TEST-CORD-02")

        self.assertEqual(res.visual_inspection_status, VisualInspectionStatus.CONFIRMED_FAULT)
        self.assertEqual(res.visual_inspection_status.value, "POTENTIAL FAULT DETECTED")
        self.assertEqual(res.overall_severity, SeverityLevel.CRITICAL)
        self.assertEqual(res.status_display_label, "Potential Cord Damage Detected")
        self.assertTrue(any("Remove from Service" in s.title for s in res.recommended_sops))
        self.assertTrue(any("Lockout" in s.title or "LOTO" in s.title for s in res.recommended_sops))

        # 3-part evidence breakdown
        self.assertTrue(len(res.visible_evidence) > 5)
        self.assertTrue(len(res.possible_explanation) > 5)
        self.assertTrue(len(res.unverified_condition) > 5)

    # -----------------------------------------------------------------
    # Scenario 3: Damaged plug/connector
    # -----------------------------------------------------------------
    def test_scenario3_damaged_plug_connector(self):
        """Scenario 3: Broken plug triggers potential fault, HIGH severity (CRITICAL under rain)."""
        cv_res = {
            "status": "success",
            "condition": "broken",
            "condition_confidence": 0.88,
            "component": "plug",
            "component_confidence": 0.90,
            "is_fault_detected": True
        }
        # Clear weather -> HIGH severity
        ctx_dry = ChargerContext(weather_condition="Clear / Dry")
        res_dry = self.engine.diagnose_from_cv(cv_res, context=ctx_dry)
        self.assertEqual(res_dry.visual_inspection_status, VisualInspectionStatus.CONFIRMED_FAULT)
        self.assertEqual(res_dry.overall_severity, SeverityLevel.HIGH)
        self.assertEqual(res_dry.status_display_label, "Potential Plug Damage Detected")
        self.assertFalse(res_dry.weather_escalated)

        # Monsoon Rain -> Escalated to CRITICAL with 2h SLA
        ctx_rain = ChargerContext(weather_condition="Monsoon Rain / Wet Ground")
        res_rain = self.engine.diagnose_from_cv(cv_res, context=ctx_rain)
        self.assertEqual(res_rain.overall_severity, SeverityLevel.CRITICAL)
        self.assertEqual(res_rain.maintenance_priority, MaintenancePriority.IMMEDIATE)
        self.assertEqual(res_rain.dispatch_sla_hours, 2)
        self.assertTrue(res_rain.weather_escalated)

    # -----------------------------------------------------------------
    # Scenario 4: Shattered screen incorrectly predicted healthy (illuminate display false negative)
    # -----------------------------------------------------------------
    def test_scenario4_shattered_screen_false_negative_handled(self):
        """Scenario 4: Shattered screen with illuminated graphics causing low-confidence healthy prediction."""
        # 4A: Automated screening with low-confidence boundary decision (e.g. 51.08% healthy)
        cv_res_boundary = {
            "status": "success",
            "condition": "healthy",
            "condition_confidence": 0.5108,  # Near boundary, below 0.65 threshold
            "component": "screen",
            "component_confidence": 0.70,
            "is_fault_detected": False
        }
        res_boundary = self.engine.diagnose_from_cv(cv_res_boundary)
        # Must NOT be declared NO VISIBLE DEFECT DETECTED
        self.assertNotEqual(res_boundary.visual_inspection_status, VisualInspectionStatus.NO_DEFECT_DETECTED)
        self.assertEqual(res_boundary.visual_inspection_status, VisualInspectionStatus.POTENTIAL_FAULT)
        self.assertEqual(res_boundary.status_display_label, "Low Confidence — Verification Required")
        self.assertEqual(res_boundary.overall_severity, SeverityLevel.UNCERTAIN)

        # 4B: Inspector observes visible screen fracture while model predicted healthy
        ctx_disagree = ChargerContext(
            inspector_visual_finding="Visible Damage Observed (Shattered / Fractured Display Glass)"
        )
        res_disagree = self.engine.diagnose_from_cv(cv_res_boundary, context=ctx_disagree)
        self.assertEqual(res_disagree.visual_inspection_status, VisualInspectionStatus.DISAGREEMENT)
        self.assertEqual(
            res_disagree.status_display_label,
            "Visible Damage Observed — Model Classification Disagrees; Manual Verification Required"
        )
        self.assertEqual(res_disagree.overall_severity, SeverityLevel.MEDIUM)
        self.assertTrue(res_disagree.classification_disagreement)
        self.assertIn("Shattered / Fractured", res_disagree.visible_evidence)

    # -----------------------------------------------------------------
    # Scenario 5: Healthy image with a low prediction score
    # -----------------------------------------------------------------
    def test_scenario5_healthy_image_with_low_prediction_score(self):
        """Scenario 5: Healthy prediction with score below 0.65 threshold must require verification."""
        cv_res = {
            "status": "success",
            "condition": "healthy",
            "condition_confidence": 0.58,  # Low score
            "component": "charger",
            "component_confidence": 0.75,
            "is_fault_detected": False
        }
        res = self.engine.diagnose_from_cv(cv_res)
        self.assertEqual(res.visual_inspection_status, VisualInspectionStatus.POTENTIAL_FAULT)
        self.assertEqual(res.status_display_label, "Low Confidence — Verification Required")
        self.assertEqual(res.overall_severity, SeverityLevel.UNCERTAIN)
        self.assertEqual(res.maintenance_priority, MaintenancePriority.PENDING_VERIFICATION)
        self.assertIn("Inconclusive visual screening", res.visible_evidence)

    # -----------------------------------------------------------------
    # Scenario 6: Missing, unavailable, and simulated telemetry
    # -----------------------------------------------------------------
    def test_scenario6_missing_unavailable_and_simulated_telemetry(self):
        """Scenario 6: Missing telemetry is unavailable; simulated telemetry is clearly distinguished."""
        # 6A: Missing / Disconnected telemetry
        ctx_none = ChargerContext(has_ocpp_telemetry=False)
        op_none = StatusAssessor.resolve_operational_status(ctx_none)
        self.assertEqual(op_none, OperationalStatus.UNAVAILABLE)
        self.assertEqual(op_none.value, "Operational status unavailable — no telemetry received")

        # 6B: Simulated telemetry
        ctx_sim = ChargerContext(
            has_ocpp_telemetry=True,
            is_telemetry_simulated=True,
            ocpp_state_raw="Operative"
        )
        op_sim = StatusAssessor.resolve_operational_status(ctx_sim)
        self.assertEqual(op_sim, OperationalStatus.SIMULATED)
        self.assertIn("Simulated Telemetry", op_sim.value)

    # -----------------------------------------------------------------
    # Scenario 7: Missing station configuration
    # -----------------------------------------------------------------
    def test_scenario7_missing_station_configuration(self):
        """Scenario 7: When station metadata is missing, display 'Not provided' without inventing values."""
        ctx_blank = ChargerContext()
        cv_res = {
            "status": "success",
            "condition": "healthy",
            "condition_confidence": 0.95,
            "component": "cord",
            "component_confidence": 0.90,
            "is_fault_detected": False
        }
        res = self.engine.diagnose_from_cv(cv_res, context=ctx_blank)
        self.assertEqual(res.station_id, "Not provided")
        self.assertEqual(res.charger_type, "Not provided")
        self.assertEqual(res.weather_condition, "Not provided")

        # Verify Report generator output handles missing configuration
        md = ReportGenerator.to_markdown(res)
        self.assertIn("**Station Identifier:** `Not provided`", md)
        self.assertIn("**Hardware Architecture:** `Not provided`", md)
        self.assertIn("**Ambient Weather:** `Not provided`", md)

    # -----------------------------------------------------------------
    # Scenario 8: PDF/HTML report consistency and duplicate-text checks
    # -----------------------------------------------------------------
    def test_scenario8_report_consistency_and_duplicate_checks(self):
        """Scenario 8: Reports must be consistent across formats, with no duplicated sections."""
        cv_res = {
            "status": "success",
            "condition": "broken",
            "condition_confidence": 0.94,
            "component": "plug",
            "component_confidence": 0.88,
            "is_fault_detected": True
        }
        ctx = ChargerContext(
            station_id="EVCS-TEST-CONSISTENCY",
            charger_type="350kW Dispenser",
            weather_condition="Clear / Dry"
        )
        diag = self.engine.diagnose_from_cv(cv_res, context=ctx)

        # Verify PDF
        pdf_bytes = ReportGenerator.to_pdf(diag)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertTrue(len(pdf_bytes) > 2000)

        # Verify Markdown
        md = ReportGenerator.to_markdown(diag)
        self.assertEqual(md.count("## 1. Observable Evidence"), 1)
        self.assertEqual(md.count("## 2. Safety Hazards"), 1)
        self.assertEqual(md.count("## 3. Recommended Personal Protective Equipment"), 1)
        self.assertEqual(md.count("## 4. Standard Operating Procedures"), 1)
        self.assertEqual(md.count("## 5. Candidate Replacement Parts"), 1)
        self.assertEqual(md.count("## 6. Applicable Technical Standards"), 1)
        self.assertEqual(md.count("## 7. Grounded Engineering Narrative"), 1)
        self.assertEqual(md.count("## 8. Qualified Technician Review"), 1)
        self.assertEqual(md.count("## 9. Mandatory Safety Disclaimer"), 1)

        # Verify HTML
        html = ReportGenerator.to_html(diag)
        self.assertIn("EVCS-TEST-CONSISTENCY", html)
        self.assertIn("350kW Dispenser", html)
        self.assertEqual(html.count("<h2>1. Observable Evidence"), 1)
        self.assertEqual(html.count("<h2>4. Recommended Standard Operating Procedures"), 1)
        self.assertEqual(html.count("<h2>8. Qualified Technician Review"), 1)

    # -----------------------------------------------------------------
    # Scenario 9: Error handling on nonexistent file
    # -----------------------------------------------------------------
    def test_error_handling_nonexistent_file(self):
        """Inference error must result in INSPECTION FAILED, never green."""
        res = self.engine.diagnose_image("non_existent_image_path.jpg")
        self.assertEqual(res.visual_inspection_status, VisualInspectionStatus.INSPECTION_FAILED)
        self.assertEqual(res.overall_severity, SeverityLevel.UNCERTAIN)
        self.assertEqual(res.maintenance_priority, MaintenancePriority.PENDING_VERIFICATION)

    # -----------------------------------------------------------------
    # Scenario 10: Technician Q&A assistant guardrails
    # -----------------------------------------------------------------
    def test_technician_qa_assistant(self):
        """Technician Q&A assistant must enforce safety directives and boundary notices."""
        cv_res = {
            "status": "success",
            "condition": "broken",
            "condition_confidence": 0.96,
            "component": "cord",
            "component_confidence": 0.92,
            "is_fault_detected": True
        }
        diag = self.engine.diagnose_from_cv(cv_res, station_id="EVCS-TEST-QA")

        ans_safe = self.engine.ask(diag, "Is it safe to charge a vehicle?")
        self.assertIn("DO NOT OPERATE", ans_safe)
        self.assertIn("LOTO", ans_safe)

        ans_ppe = self.engine.ask(diag, "What PPE gloves do I need?")
        self.assertIn("PPE", ans_ppe)

        ans_parts = self.engine.ask(diag, "What spare parts can I order?")
        self.assertIn("cable assembly", ans_parts.lower())


    # -----------------------------------------------------------------
    # Scenario 11: Screen-specific safety rule (High-confidence healthy screen)
    # -----------------------------------------------------------------
    def test_scenario11_screen_healthy_high_confidence_safety_rule(self):
        """Screen classified healthy with high confidence (>=0.65) must require manual verification."""
        cv_res = {
            "status": "success",
            "condition": "healthy",
            "condition_confidence": 0.98,
            "component": "screen",
            "component_confidence": 0.95,
            "is_fault_detected": False,
        }
        res = self.engine.diagnose_from_cv(cv_res, station_id="EVCS-TEST-SCREEN-01")
        self.assertEqual(res.visual_inspection_status, VisualInspectionStatus.POTENTIAL_FAULT)
        self.assertEqual(res.overall_severity, SeverityLevel.UNCERTAIN)
        self.assertEqual(res.maintenance_priority, MaintenancePriority.PENDING_VERIFICATION)
        self.assertEqual(
            res.status_display_label,
            "Screen Inspection Required — AI Result Not Sufficient to Confirm Condition"
        )
        self.assertIn("cannot reliably certify screen integrity", res.safety_hazards[1].lower())
        self.assertIn("Screen Inspection Required", res.possible_explanation)

    def test_scenario12_screen_broken_high_confidence(self):
        """Broken screen with high confidence retains CONFIRMED_FAULT and SCHEDULED priority."""
        cv_res = {
            "status": "success",
            "condition": "broken",
            "condition_confidence": 0.91,
            "component": "screen",
            "component_confidence": 0.85,
            "is_fault_detected": True,
        }
        res = self.engine.diagnose_from_cv(cv_res, station_id="EVCS-TEST-SCREEN-02")
        self.assertEqual(res.visual_inspection_status, VisualInspectionStatus.CONFIRMED_FAULT)
        self.assertEqual(res.overall_severity, SeverityLevel.MEDIUM)
        self.assertEqual(res.maintenance_priority, MaintenancePriority.SCHEDULED)
        self.assertEqual(res.status_display_label, "Potential Screen Damage Detected")

    def test_scenario13_charger_healthy_high_confidence_unchanged(self):
        """Charger enclosure classified healthy with high confidence remains NO_DEFECT_DETECTED."""
        cv_res = {
            "status": "success",
            "condition": "healthy",
            "condition_confidence": 0.99,
            "component": "charger",
            "component_confidence": 0.96,
            "is_fault_detected": False,
        }
        res = self.engine.diagnose_from_cv(cv_res, station_id="EVCS-TEST-CHARGER-02")
        self.assertEqual(res.visual_inspection_status, VisualInspectionStatus.NO_DEFECT_DETECTED)
        self.assertEqual(res.overall_severity, SeverityLevel.NONE_DETECTED)
        self.assertEqual(res.maintenance_priority, MaintenancePriority.ROUTINE)
        self.assertEqual(res.status_display_label, "No Visible Defect Detected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
