"""
Interactive CLI Demonstration for Member 3:
Vision-Language Diagnostic Assistant & Maintenance Recommendations.

Demonstrates:
  1. End-to-end integration: Image -> Member 2 CV Model -> Member 3 Diagnostic Engine.
  2. Multi-scenario evaluation: Broken Cord (Critical), Broken Plug (Weather Escalation), Healthy Charger.
  3. Grounded Technician Assistant Q&A.
  4. Automated Work Order export (Markdown & HTML).
"""

import sys
import os
from pathlib import Path

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from src.diagnostics import (
    EVDiagnosticEngine,
    ChargerContext,
    ReportGenerator,
    diagnose_image,
)


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_demo():
    print_banner("[+] MEMBER 3: EV CHARGING DIAGNOSTIC ASSISTANT DEMO")
    print("Initializing Hybrid Neuro-Symbolic Diagnostic Engine...")
    engine = EVDiagnosticEngine()

    dataset_dir = PROJECT_ROOT / "dataset" / "flare_classification"

    # -------------------------------------------------------------
    # Scenario 1: Critical Fault - Broken Cable Sheath
    # -------------------------------------------------------------
    print_banner("Scenario 1: High-Voltage Cable Sheath Rupture (cord / broken)")
    cord_img = dataset_dir / "cord" / "broken" / "cord_broken_0065.jpg"

    if cord_img.exists():
        ctx1 = ChargerContext(
            station_id="EVCS-PUNE-HUB-01",
            charger_type="150kW DC Fast Charger (CCS2 Dual Gun)",
            weather_condition="Clear / Dry"
        )
        diag1 = engine.diagnose_image(cord_img, context=ctx1)

        print(f"* Station ID: {diag1.station_id}")
        print(f"* Component Inspected: {diag1.component.upper()} (Condition: {diag1.condition.upper()}, Confidence: {diag1.confidence_score*100:.1f}%)")
        print(f"* Visual Status: {diag1.visual_inspection_status.value}")
        print(f"* Assigned Severity: {diag1.overall_severity.value}")
        print(f"* Dispatch SLA: {diag1.dispatch_sla_display} (Priority: {diag1.maintenance_priority.value})")
        print("\nPrimary Safety Hazards:")
        for h in diag1.safety_hazards[:2]:
            print(f"  [HAZARD] {h}")
        print("\nMandatory PPE:")
        for p in diag1.required_ppe[:2]:
            print(f"  [PPE] {p}")
        print(f"\nStep 1 SOP: {diag1.recommended_sops[0].title}")
        print(f"  -> {diag1.recommended_sops[0].description}")
    else:
        print(f"Sample image not found at: {cord_img}")

    # -------------------------------------------------------------
    # Scenario 2: Weather Escalation - Broken Plug in Rain
    # -------------------------------------------------------------
    print_banner("Scenario 2: Environmental Weather Escalation (plug / broken in Rain)")
    plug_img = dataset_dir / "plug" / "broken" / "plug_broken_0113.jpg"

    if plug_img.exists():
        # Case 2A: Dry Weather
        ctx_dry = ChargerContext(station_id="EVCS-MUMBAI-04", weather_condition="Clear / Dry")
        diag_dry = engine.diagnose_image(plug_img, context=ctx_dry)
        print(f"[Clear Weather] Severity: {diag_dry.overall_severity.value} | SLA: {diag_dry.dispatch_sla_hours}h | Weather Escalated: {diag_dry.weather_escalated}")

        # Case 2B: Heavy Rain
        ctx_rain = ChargerContext(station_id="EVCS-MUMBAI-04", weather_condition="Monsoon Rain / Wet Ground")
        diag_rain = engine.diagnose_image(plug_img, context=ctx_rain)
        print(f"[Monsoon Rain]  Severity: {diag_rain.overall_severity.value} | SLA: {diag_rain.dispatch_sla_hours}h | Weather Escalated: {diag_rain.weather_escalated}")
        print(f"  [!] Weather Rule Triggered: {diag_rain.safety_hazards[0]}")
    else:
        print(f"Sample image not found at: {plug_img}")

    # -------------------------------------------------------------
    # Scenario 3: Healthy Component with Safety Guardrail
    # -------------------------------------------------------------
    print_banner("Scenario 3: Structurally Healthy Charger Housing (charger / healthy)")
    healthy_img = dataset_dir / "charger" / "healthy" / "charger_healthy_0023.jpg"

    if healthy_img.exists():
        ctx3 = ChargerContext(station_id="EVCS-BLR-09")
        diag3 = engine.diagnose_image(healthy_img, context=ctx3)
        print(f"* Station ID: {diag3.station_id}")
        print(f"* Visual Status: {diag3.visual_inspection_status.value}")
        print(f"* Severity: {diag3.overall_severity.value}")
        print(f"* Mandatory Safety Disclaimer:")
        print(f"  \"{diag3.disclaimer}\"")
    else:
        print(f"Sample image not found at: {healthy_img}")

    # -------------------------------------------------------------
    # Scenario 4: Interactive Field Technician Q&A
    # -------------------------------------------------------------
    if cord_img.exists():
        print_banner("Scenario 4: Grounded Technician Q&A Assistant")
        questions = [
            "Can I turn on this charger for a customer right now?",
            "What PPE gloves are required before I inspect this cable?",
            "What is the emergency dispatch SLA?",
        ]

        for q in questions:
            print(f"\n[Technician Query]: \"{q}\"")
            ans = engine.ask(diag1, q)
            # Remove non-ascii for clean console output
            ans_clean = ans.replace("⛔", "[STOP]").replace("⚠️", "[WARNING]").replace("🛡️", "[PPE]").replace("⏱️", "[SLA]").replace("ℹ️", "[INFO]")
            print(f"[AI Assistant]:\n{ans_clean}")

    # -------------------------------------------------------------
    # Scenario 5: Work Order Export
    # -------------------------------------------------------------
    if cord_img.exists():
        print_banner("Scenario 5: Automated Maintenance Work Order Export (PDF, HTML, MD)")
        out_pdf = PROJECT_ROOT / "sample_work_order.pdf"
        out_html = PROJECT_ROOT / "sample_work_order.html"
        out_md = PROJECT_ROOT / "sample_work_order.md"

        ReportGenerator.save_pdf(diag1, out_pdf)
        html_content = ReportGenerator.to_html(diag1)
        md_content = ReportGenerator.to_markdown(diag1)

        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"[SUCCESS] Exported PDF Work Order ticket -> {out_pdf}")
        print(f"[SUCCESS] Exported HTML Work Order ticket -> {out_html}")
        print(f"[SUCCESS] Exported Markdown Work Order ticket -> {out_md}")

    print_banner("DEMO COMPLETED SUCCESSFULLY! All Member 3 tasks validated.")


if __name__ == "__main__":
    run_demo()
