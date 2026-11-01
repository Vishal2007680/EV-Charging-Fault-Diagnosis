"""
Automated Work Order and Inspection Report Generator for Member 3.
Exports certified maintenance tickets in Markdown, HTML, and JSON formats.
"""

from datetime import datetime
import json
from .schema import DiagnosticResult, SeverityLevel, VisualInspectionStatus


class ReportGenerator:
    """Generates certified EV Maintenance Work Orders and Inspection Reports."""

    @staticmethod
    def to_markdown(result: DiagnosticResult) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        status_badge = {
            VisualInspectionStatus.CONFIRMED_FAULT: "🔴 **CONFIRMED FAULT DETECTED**",
            VisualInspectionStatus.POTENTIAL_FAULT: "🟠 **POTENTIAL FAULT — VERIFICATION REQUIRED**",
            VisualInspectionStatus.NO_DEFECT_DETECTED: "🔵 **NO SUPPORTED DEFECT DETECTED**",
            VisualInspectionStatus.INSPECTION_FAILED: "⚫ **INSPECTION FAILED**",
        }.get(result.visual_inspection_status, result.visual_inspection_status.value)

        sev_badge = {
            SeverityLevel.CRITICAL: "🚨 `CRITICAL`",
            SeverityLevel.HIGH: "⚠️ `HIGH`",
            SeverityLevel.MEDIUM: "⚡ `MEDIUM`",
            SeverityLevel.LOW: "ℹ️ `LOW`",
            SeverityLevel.UNCERTAIN: "❓ `UNCERTAIN`",
            SeverityLevel.NONE_DETECTED: "✅ `NONE_DETECTED`",
        }.get(result.overall_severity, f"`{result.overall_severity.value}`")

        weather_notice = " *(Escalated due to ambient rain/moisture)*" if result.weather_escalated else ""

        md = [
            "# ⚡ EV CHARGING STATION VISUAL INSPECTION WORK ORDER",
            "**Document Title:** EV CHARGING STATION VISUAL INSPECTION REPORT  ",
            f"**Station Identifier:** `{result.station_id}`  ",
            f"**Generated At:** {date_str}  ",
            f"**Visual Inspection Status:** {status_badge}  ",
            f"**Assigned Severity Level:** {sev_badge}{weather_notice}  ",
            f"**Operational Telemetry:** `{result.operational_status.value}`  ",
            f"**Maintenance Priority:** `{result.maintenance_priority.value}`  ",
            f"**Emergency Dispatch SLA:** **{result.dispatch_sla_hours or 'N/A'} Hours**  ",
            f"**Inspected Component:** `{result.component or 'General'}`  ",
            f"**AI Condition Confidence:** **{result.confidence_score*100:.1f}%**  ",
            "",
            "---",
            "## 1. Safety Hazards & Warnings",
        ]

        for h in result.safety_hazards:
            md.append(f"- ⚠️ {h}")

        md.extend([
            "",
            "## 2. Mandatory Personal Protective Equipment (PPE)",
        ])
        for p in result.required_ppe:
            md.append(f"- 🛡️ {p}")

        md.extend([
            "",
            "## 3. Observable Visual Evidence & Engineering Findings",
        ])
        for exp in result.fault_explanations:
            md.append(f"- {exp}")

        md.extend([
            "",
            "## 4. Standard Operating Procedures (SOP)",
        ])
        for sop in result.recommended_sops:
            critical_badge = "🚨 **[SAFETY CRITICAL]**" if sop.is_safety_critical else f"ℹ️ [{sop.action_type.upper()}]"
            md.append(f"### Step {sop.step_number}: {sop.title} {critical_badge}")
            md.append(f"{sop.description}\n")

        md.extend([
            "## 5. Candidate Replacement Parts",
        ])
        if result.replacement_parts:
            for part in result.replacement_parts:
                md.append(f"- 🔧 {part}")
        else:
            md.append("- No replacement parts currently required.")

        md.extend([
            "",
            "## 6. Referenced Technical Standards (Advisory)",
        ])
        for std in result.standards_referenced:
            md.append(f"- 📜 {std}")

        md.extend([
            "",
            "## 7. Grounded Engineering Narrative",
            "```text",
            result.llm_reasoning_summary or "No summary generated.",
            "```",
            "",
            "> ⚠️ **SAFETY DISCLAIMER:** " + result.disclaimer
        ])

        return "\n".join(md)

    @staticmethod
    def to_html(result: DiagnosticResult) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        sev_colors = {
            SeverityLevel.CRITICAL: ("#ffebee", "#c62828", "#b71c1c"),
            SeverityLevel.HIGH: ("#fff3e0", "#ef6c00", "#e65100"),
            SeverityLevel.MEDIUM: ("#fffde7", "#f57f17", "#f57f17"),
            SeverityLevel.LOW: ("#e3f2fd", "#1565c0", "#0d47a1"),
            SeverityLevel.UNCERTAIN: ("#f3e5f5", "#7b1fa2", "#4a148c"),
            SeverityLevel.NONE_DETECTED: ("#e8f5e9", "#2e7d32", "#1b5e20"),
        }
        bg_col, txt_col, border_col = sev_colors.get(
            result.overall_severity, ("#fafafa", "#333333", "#999999")
        )

        hazards_html = "".join([f"<li>⚠️ {h}</li>" for h in result.safety_hazards])
        ppe_html = "".join([f"<li>🛡️ {p}</li>" for p in result.required_ppe])
        explanations_html = "".join([f"<li>{e}</li>" for e in result.fault_explanations])
        parts_html = "".join([f"<li>🔧 {p}</li>" for p in result.replacement_parts]) or "<li>No replacement parts currently required</li>"
        stds_html = "".join([f"<li>📜 {s}</li>" for s in result.standards_referenced])

        sops_html = ""
        for sop in result.recommended_sops:
            crit_badge = "<span style='color:#c62828; font-weight:bold; margin-left:8px;'>[SAFETY CRITICAL]</span>" if sop.is_safety_critical else f"<span style='color:#555; margin-left:8px;'>[{sop.action_type.upper()}]</span>"
            sops_html += f"""
            <div style="margin-bottom: 12px; padding: 12px; background: #ffffff; border-left: 4px solid {border_col}; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <strong style="color: #111;">Step {sop.step_number}: {sop.title}</strong> {crit_badge}<br>
                <div style="margin-top: 6px; font-size: 0.95em; color: #444; line-height: 1.5;">{sop.description}</div>
            </div>
            """

        weather_banner = ""
        if result.weather_escalated:
            weather_banner = """
            <div style="background: #ffebee; border: 2px solid #c62828; color: #b71c1c; padding: 10px 14px; border-radius: 6px; margin: 12px 0; font-weight: bold;">
                🚨 WEATHER ESCALATION: Severity elevated to CRITICAL due to rain / wet ambient surface conditions!
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EV Charging Visual Inspection Report - {result.station_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 40px auto;
            max-width: 960px;
            color: #222;
            background: #f8f9fa;
            line-height: 1.6;
        }}
        .report-card {{
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 32px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }}
        .header {{
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.9em;
            background: {bg_col};
            color: {txt_col};
            border: 1px solid {border_col};
        }}
        .hazards-box {{
            background: {bg_col};
            border: 1px solid {border_col};
            border-radius: 6px;
            padding: 16px 20px;
            margin-bottom: 24px;
        }}
        .hazards-box h3 {{
            margin-top: 0;
            color: {txt_col};
        }}
        h2 {{
            color: #1a237e;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
            margin-top: 28px;
        }}
        ul {{ padding-left: 20px; }}
        li {{ margin-bottom: 6px; }}
        .summary-pre {{
            background: #263238;
            color: #eceff1;
            padding: 16px;
            border-radius: 6px;
            white-space: pre-wrap;
            font-family: "Courier New", Courier, monospace;
            font-size: 0.9em;
        }}
        .disclaimer {{
            background: #fff8e1;
            border-left: 5px solid #ffb300;
            padding: 14px 18px;
            border-radius: 4px;
            font-size: 0.9em;
            color: #6d4c41;
            margin-top: 30px;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 14px;
        }}
        .meta-item {{
            background: #f1f3f4;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <h1 style="margin:0 0 8px 0; color: #0d47a1;">⚡ EV Charging Station Visual Inspection Report</h1>
            <div style="color: #666; font-size: 0.95em;">Certified Diagnostic Ticket | Safety & Maintenance SOP Engine</div>

            <div class="meta-grid">
                <div class="meta-item"><strong>Station ID:</strong> {result.station_id}</div>
                <div class="meta-item"><strong>Timestamp:</strong> {date_str}</div>
                <div class="meta-item"><strong>Component:</strong> {result.component.upper() if result.component else 'GENERAL'}</div>
                <div class="meta-item"><strong>AI Confidence:</strong> {result.confidence_score*100:.1f}%</div>
            </div>

            <div style="margin-top: 16px;">
                <span class="badge">Visual: {result.visual_inspection_status.value}</span>
                <span class="badge" style="margin-left: 8px;">Severity: {result.overall_severity.value}</span>
                <span class="badge" style="margin-left: 8px; background: #e0f2fe; color: #0369a1; border-color: #7dd3fc;">SLA: {result.dispatch_sla_hours or 'N/A'} Hours</span>
            </div>
            <div style="margin-top: 10px; font-size: 0.9em; color: #555;">
                <strong>Operational Status:</strong> <code>{result.operational_status.value}</code> &nbsp;|&nbsp;
                <strong>Priority:</strong> {result.maintenance_priority.value}
            </div>
        </div>

        {weather_banner}

        <div class="hazards-box">
            <h3>⚠️ Safety Hazards & Compliance Directives</h3>
            <ul>{hazards_html}</ul>
        </div>

        <h2>🛡️ Required Personal Protective Equipment (PPE)</h2>
        <ul>{ppe_html}</ul>

        <h2>🔍 Observable Findings & Root-Cause Hypotheses</h2>
        <ul>{explanations_html}</ul>

        <h2>📋 Recommended Standard Operating Procedures (SOP)</h2>
        {sops_html}

        <h2>🔧 Candidate Replacement Parts</h2>
        <ul>{parts_html}</ul>

        <h2>📜 Referenced Technical Standards</h2>
        <ul>{stds_html}</ul>

        <h2>🤖 Grounded Diagnostic Narrative</h2>
        <div class="summary-pre">{result.llm_reasoning_summary or 'No summary recorded.'}</div>

        <div class="disclaimer">
            <strong>⚠️ MANDATORY SAFETY DISCLAIMER:</strong><br>
            {result.disclaimer}
        </div>
    </div>
</body>
</html>"""
        return html

    @staticmethod
    def to_json(result: DiagnosticResult) -> str:
        return json.dumps(result.to_dict(), indent=2)
