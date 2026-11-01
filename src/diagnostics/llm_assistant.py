"""
Grounded Multimodal Assistant and Technical Q&A Engine for Field Technicians.
Adheres strictly to evidence-based safety engineering:
  - Answers strictly grounded in the current visual inspection findings.
  - Formulates clear boundaries between observable evidence and unverified parameters.
  - Disallows assertions of certification or compliance.
  - Operates 100% offline deterministically, with optional live LLM API support.
  - Handles visual disagreements (damage observed vs model healthy).
"""

import os
import json
from typing import Optional, List, Dict, Any

from .schema import (
    DiagnosticResult,
    VisualInspectionStatus,
    OperationalStatus,
    SeverityLevel,
    MaintenancePriority,
    MaintenanceAction,
)


class LLMDiagnosticAssistant:
    """Provides grounded synthesis and interactive technician Q&A."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                self.client = None

    def generate_synthesis(
        self,
        station_id: Optional[str],
        visual_status: VisualInspectionStatus,
        operational_status: OperationalStatus,
        severity: SeverityLevel,
        component: Optional[str],
        condition: Optional[str],
        confidence: float,
        hazards: List[str],
        explanations: List[str],
        sops: List[MaintenanceAction],
        charger_type: Optional[str] = None,
        weather: Optional[str] = None,
        is_telemetry_simulated: bool = False,
        inspector_visual_finding: Optional[str] = None,
        classification_disagreement: bool = False
    ) -> str:
        """
        Produces a grounded, professional engineering diagnostic narrative.
        Works offline via expert rule synthesis, or via live LLM if configured.
        """
        st_id = station_id if station_id and station_id.strip() else "Not provided"
        ch_type = charger_type if charger_type and charger_type.strip() else "Not provided"
        w_cond = weather if weather and weather.strip() else "Not provided"

        op_str = operational_status.value
        if is_telemetry_simulated:
            op_str = f"{op_str} (Simulated - No live link)"

        # If live LLM is active, attempt grounded prompt generation
        if self.client is not None:
            try:
                live_prompt = f"""
                Station ID: {st_id}
                Architecture: {ch_type}
                Ambient Weather: {w_cond}
                Visual Inspection Status: {visual_status.value}
                Operational Telemetry: {op_str}
                Severity: {severity.value}
                Component: {component} (Condition: {condition}, Model Score: {confidence*100:.1f}%)

                Hazards:
                {chr(10).join(['- ' + h for h in hazards])}

                SOPs:
                {chr(10).join([f"- Step {s.step_number}: {s.title} ({s.description})" for s in sops])}

                Write an authoritative 3-paragraph diagnostic summary for the field maintenance log.
                Strictly separate observable physical facts from hypotheses. Emphasize that visual inspection does not verify internal electrical condition.
                """
                if hasattr(self.client, "generate_content"):
                    resp = self.client.generate_content(live_prompt)
                    if resp and resp.text:
                        return resp.text.strip()
            except Exception:
                pass  # Fall back to deterministic synthesis

        # Deterministic Grounded Expert Synthesis (100% offline)
        narrative = [
            "=== AI-ASSISTED VISUAL INSPECTION SUMMARY (RESEARCH PROTOTYPE) ===",
            f"Station Identifier: {st_id.upper()}",
            f"Hardware Architecture: {ch_type} | Ambient Weather: {w_cond}",
            f"Visual Inspection Status: {visual_status.value}",
            f"Operational Telemetry: {op_str}",
            f"Assigned Severity Level: {severity.value}",
        ]

        if classification_disagreement:
            narrative.append("")
            narrative.append(f"⚠️ CLASSIFICATION DISAGREEMENT NOTICE: Visible damage observed ('{inspector_visual_finding}'), while classifier predicted HEALTHY.")

        narrative.append("")
        narrative.append("1. OBSERVABLE EVIDENCE & PRELIMINARY HYPOTHESES:")

        if explanations:
            for exp in explanations:
                narrative.append(f"  • {exp}")
        else:
            narrative.append("  • No visible defect identified from the provided visual frame.")

        narrative.append("")
        narrative.append("2. SAFETY & RELIABILITY HAZARDS:")
        for h in hazards:
            narrative.append(f"  • {h}")

        narrative.append("")
        narrative.append("3. RECOMMENDED STANDARD OPERATING PROCEDURES (SOP):")
        if sops:
            for s in sops:
                crit_tag = " [SAFETY CRITICAL]" if s.is_safety_critical else f" [{s.action_type.upper()}]"
                narrative.append(f"  • Step {s.step_number} ({s.title}){crit_tag}: {s.description}")
        else:
            narrative.append("  • Follow regular scheduled preventive maintenance per manufacturer guidelines.")

        narrative.append("")
        narrative.append("4. IMPORTANT ENGINEERING SAFETY BOUNDARIES:")
        narrative.append(
            "  • This evaluation is an AI-assisted visual screening tool. Optical inspection cannot "
            "verify contact resistance, internal conductor insulation integrity, internal heating, "
            "or electrical safety compliance. Physical testing by qualified personnel using approved "
            "procedures is required prior to equipment re-energization or return to service."
        )

        return "\n".join(narrative)

    def answer_technician_query(
        self,
        diagnostic: DiagnosticResult,
        user_question: str
    ) -> str:
        """
        Interactive Q&A assistant for field technicians during maintenance.
        Enforces strict safety guardrails and distinguishes observations from unverified parameters.
        """
        q = user_question.lower()

        # Safety & Operation Queries
        if any(k in q for k in ["safe", "operate", "use", "turn on", "charge", "charging", "re-energize", "touch"]):
            if diagnostic.visual_inspection_status in [
                VisualInspectionStatus.CONFIRMED_FAULT,
                VisualInspectionStatus.DISAGREEMENT,
                "POTENTIAL FAULT DETECTED",
                "VISIBLE DAMAGE OBSERVED — MODEL CLASSIFICATION DISAGREES"
            ]:
                finding_desc = diagnostic.inspector_visual_finding or "potential damage"
                return (
                    f"⛔ **POTENTIAL FAULT DETECTED / DAMAGE OBSERVED — DO NOT OPERATE.**\n\n"
                    f"Visual screening or physical observation identified defect ({finding_desc}) on the **{diagnostic.component}** "
                    f"with an assigned severity of **{diagnostic.overall_severity.value}**.\n\n"
                    f"**Identified Risk:** {diagnostic.safety_hazards[0] if diagnostic.safety_hazards else 'Potential electrical safety hazard'}.\n\n"
                    f"**Precautionary Action:** Keep the station de-energized. Qualified personnel must follow the "
                    f"installation's approved Lockout/Tagout (LOTO) and zero-energy verification procedure before any physical touch or maintenance."
                )
            elif diagnostic.visual_inspection_status == VisualInspectionStatus.POTENTIAL_FAULT:
                return (
                    f"⚠️ **CONDITIONAL — ON-SITE VERIFICATION REQUIRED.**\n\n"
                    f"The system flagged **{diagnostic.visual_inspection_status.value}** with "
                    f"model score of {diagnostic.confidence_score*100:.1f}%.\n\n"
                    f"Do NOT approve customer charging until a qualified technician has visually inspected "
                    f"and physically verified the {diagnostic.component}."
                )
            elif diagnostic.visual_inspection_status == VisualInspectionStatus.INSPECTION_FAILED:
                return (
                    "⛔ **CANNOT CONFIRM SAFETY — INSPECTION INCONCLUSIVE.**\n\n"
                    "The automated camera inspection failed or was inconclusive. "
                    "Never assume equipment is operational when diagnosis fails. Perform manual on-site inspection."
                )
            else:
                return (
                    f"ℹ️ **NO VISIBLE SURFACE DEFECT DETECTED ON {diagnostic.component.upper() if diagnostic.component else 'EQUIPMENT'}.**\n\n"
                    f"Exterior physical structure appears intact from the provided image ({diagnostic.confidence_score*100:.1f}% model score).\n\n"
                    f"**Important Safety Warning:** Optical inspection does not verify internal electrical contact resistance, "
                    f"insulation integrity, or earth continuity. Confirm operational telemetry ({diagnostic.operational_status.value}) "
                    f"and complete approved pre-commissioning checks before declaring station operational."
                )

        # PPE Queries
        if any(k in q for k in ["ppe", "glove", "gear", "wear", "suit", "protection", "glasses", "boot"]):
            ppe_list = "\n".join([f"  • {p}" for p in diagnostic.required_ppe])
            return (
                f"🛡️ **Recommended Personal Protective Equipment (Station {diagnostic.station_id}):**\n\n"
                f"{ppe_list}\n\n"
                f"*Note: Electrical PPE recommendations are advisory and must be confirmed by qualified personnel based on site hazard assessment and applicable requirements.*"
            )

        # SOP / Procedure Queries
        if any(k in q for k in ["sop", "procedure", "step", "how to fix", "repair", "replace", "action"]):
            if diagnostic.recommended_sops:
                sops_formatted = []
                for s in diagnostic.recommended_sops:
                    crit = "🚨 **[SAFETY CRITICAL]**" if s.is_safety_critical else f"ℹ️ [{s.action_type.upper()}]"
                    sops_formatted.append(f"**Step {s.step_number}: {s.title}** {crit}\n{s.description}")
                return (
                    f"📋 **Recommended Standard Operating Procedures (SOP):**\n\n"
                    + "\n\n".join(sops_formatted)
                    + "\n\n*Notice: Follow the manufacturer's approved service procedure; equipment-specific torque, test limits, and instructions must be taken from official equipment documentation.*"
                )
            return "No active repair procedures identified from visual imagery. Follow the manufacturer's regular scheduled preventive maintenance procedure."

        # Spare Parts Queries
        if any(k in q for k in ["part", "replace", "spare", "bill of material", "order", "bom"]):
            if diagnostic.replacement_parts:
                parts_list = "\n".join([f"  • {p}" for p in diagnostic.replacement_parts])
                return (
                    f"🔧 **Candidate Replacement Parts (Station {diagnostic.station_id}):**\n\n"
                    f"{parts_list}\n\n"
                    f"*Important: Parts must be verified for compatibility with the specific charger model and connector standard. CCS2, CHAdeMO, and Type 2 components are not interchangeable.*"
                )
            return "ℹ️ No replacement part identified from available evidence."

        # Dispatch SLA / Urgency Queries
        if any(k in q for k in ["sla", "time", "urgent", "priority", "dispatch", "deadline", "how fast"]):
            if diagnostic.dispatch_sla_hours:
                return (
                    f"⏱️ **Dispatch SLA:** **{diagnostic.dispatch_sla_hours} Hours** (Configured / Weather Escalated)\n\n"
                    f"**Maintenance Priority:** {diagnostic.maintenance_priority.value}\n"
                    f"**Severity Level:** {diagnostic.overall_severity.value}"
                )
            return (
                f"⏱️ **Dispatch SLA:** **Not assigned**\n\n"
                f"**Maintenance Priority:** {diagnostic.maintenance_priority.value}\n"
                f"**Severity Level:** {diagnostic.overall_severity.value}\n\n"
                f"*No emergency dispatch SLA is configured. Follow standard organizational dispatch procedures.*"
            )

        # Standards Queries
        if any(k in q for k in ["standard", "compliance", "iec", "nfpa", "code", "regulation"]):
            standards_list = "\n".join([f"  • {s}" for s in diagnostic.standards_referenced])
            return (
                f"📜 **Applicable Technical References (Advisory):**\n\n"
                f"{standards_list}\n\n"
                f"*Notice: These references provide relevant technical context for qualified personnel. The software does not provide standards certification.*"
            )

        # Weather Queries
        if any(k in q for k in ["weather", "rain", "wet", "water", "heat"]):
            if diagnostic.weather_escalated:
                return (
                    "🚨 **WEATHER ESCALATION ACTIVE:** Rain / wet surface conditions in conjunction with suspected coupler or cable damage "
                    "escalate the safety severity to CRITICAL due to heightened risk of dielectric breakdown and ground fault."
                )
            return "Ambient weather conditions are accounted for in the diagnostic severity evaluation. High humidity or rain elevates electrical risk."

        # General Fallback
        return (
            f"Advisory for Station **{diagnostic.station_id}**:\n\n"
            f"- **Visual Status:** {diagnostic.visual_inspection_status.value}\n"
            f"- **Severity:** {diagnostic.overall_severity.value}\n"
            f"- **Maintenance Priority:** {diagnostic.maintenance_priority.value}\n\n"
            f"For specific directives, ask about 'SOP', 'safety', 'PPE', 'parts', or 'standards'."
        )
