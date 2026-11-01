"""
Automated Work Order and Inspection Report Generator.
Exports standardized maintenance tickets in PDF, HTML, Markdown, and JSON formats.
Follows evidence-based research prototype standards:
  - Title: AI-Assisted Visual Inspection Report — Research Prototype
  - Clean section numbering (Sections 1 through 9) without skipped numbers
  - Strict separation of visible evidence, possible explanations, and unverified parameters
  - Verified station metadata (architecture, weather, telemetry status) without invented values
  - Clear classification disagreement and weather escalation callout banners
  - Dynamic aspect-ratio image embedding in PDF and HTML
  - Blank qualified technician sign-off block (does not claim pre-existing sign-off)
"""

import os
import io
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from PIL import Image as PILImage

from .schema import DiagnosticResult, SeverityLevel, VisualInspectionStatus

# ReportLab imports for professional PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page counts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header
        self.drawString(36, 808, "EV Charging Station Visual Inspection — Research Prototype")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 802, 559, 802)

        # Running Footer
        self.line(36, 36, 559, 36)
        self.drawString(36, 24, "AI-Assisted Visual Inspection Report — Not an Electrical Safety Certification")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(559, 24, page_text)
        self.restoreState()


class ReportGenerator:
    """Generates standardized EV Maintenance Work Orders and Inspection Reports."""

    @classmethod
    def to_pdf(cls, result: DiagnosticResult, image_path: Optional[str] = None) -> bytes:
        """Generates a multi-page, publication-quality A4 PDF work order."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=48,
            bottomMargin=48
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        style_title = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=3,
        )
        style_subtitle = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#475569"),
            spaceAfter=8,
        )
        style_h2 = ParagraphStyle(
            "SectionH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        )
        style_body = ParagraphStyle(
            "BodyTextCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#334155"),
        )
        style_bold_body = ParagraphStyle(
            "BoldBodyCustom",
            parent=style_body,
            fontName="Helvetica-Bold",
        )
        style_disclaimer = ParagraphStyle(
            "DisclaimerText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#92400e"),
        )
        style_alert = ParagraphStyle(
            "AlertText",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#b91c1c"),
        )

        story = []

        # Header Title
        story.append(Paragraph("⚡ AI-Assisted Visual Inspection Report — Research Prototype", style_title))
        story.append(Paragraph("Preliminary Optical Screening & Grounded Maintenance Advisory | Research Prototype Only", style_subtitle))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=8))

        # Metadata Table
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        meta = result.cv_metadata or {}
        latency_str = f"{meta.get('execution_time_ms', 'N/A')} ms" if meta.get('execution_time_ms') else "N/A"
        sla_display = result.dispatch_sla_display

        op_status_str = result.operational_status.value
        if result.is_telemetry_simulated:
            op_status_str += " [SIMULATED]"

        meta_data = [
            [
                Paragraph("<b>Report Identifier:</b>", style_body), Paragraph(result.report_id, style_body),
                Paragraph("<b>Station Identifier:</b>", style_body), Paragraph(result.station_id, style_body),
            ],
            [
                Paragraph("<b>Hardware Architecture:</b>", style_body), Paragraph(result.charger_type, style_body),
                Paragraph("<b>Ambient Weather:</b>", style_body), Paragraph(result.weather_condition, style_body),
            ],
            [
                Paragraph("<b>Inspection Date:</b>", style_body), Paragraph(date_str, style_body),
                Paragraph("<b>Inspected Component:</b>", style_body), Paragraph((result.component or "General").upper(), style_bold_body),
            ],
            [
                Paragraph("<b>Visual Status:</b>", style_body), Paragraph(result.visual_inspection_status.value, style_bold_body),
                Paragraph("<b>Safety Severity:</b>", style_body), Paragraph(result.overall_severity.value, style_bold_body),
            ],
            [
                Paragraph("<b>AI Model Score:</b>", style_body), Paragraph(f"{result.confidence_score*100:.1f}%", style_body),
                Paragraph("<b>Dispatch SLA:</b>", style_body), Paragraph(sla_display, style_body),
            ],
            [
                Paragraph("<b>Operational Telemetry:</b>", style_body), Paragraph(op_status_str, style_body),
                Paragraph("<b>Inference Latency:</b>", style_body), Paragraph(latency_str, style_body),
            ],
        ]

        meta_table = Table(meta_data, colWidths=[110, 150, 115, 145])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 8))

        # Classification Disagreement banner if active
        if result.classification_disagreement:
            disagree_box = [
                [Paragraph(f"<b>⚠️ CLASSIFICATION DISAGREEMENT NOTICE:</b> Physical visual findings indicate potential defect ('{result.inspector_visual_finding}'), while the 2D classification model predicted HEALTHY ({result.confidence_score*100:.1f}%). Both observations are preserved; manual physical verification is required.", style_alert)]
            ]
            d_table = Table(disagree_box, colWidths=[520])
            d_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#ef4444")),
                ('PADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(d_table)
            story.append(Spacer(1, 6))

        # Weather escalation callout if active
        if result.weather_escalated:
            weather_box = [
                [Paragraph("<b>🚨 WEATHER ESCALATION ACTIVE:</b> Ambient wet / rain conditions present with suspected high-voltage defect. Severity elevated to CRITICAL.", style_disclaimer)]
            ]
            w_table = Table(weather_box, colWidths=[520])
            w_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fff7ed")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#ea580c")),
                ('PADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(w_table)
            story.append(Spacer(1, 6))

        # Embedded Inspection Image (aspect-ratio preserved)
        effective_img_path = image_path or result.image_path
        if effective_img_path and os.path.exists(effective_img_path):
            try:
                with PILImage.open(effective_img_path) as pil_img:
                    orig_w, orig_h = pil_img.size
                max_w, max_h = 360.0, 150.0
                ratio = min(max_w / orig_w, max_h / orig_h, 1.0)
                final_w, final_h = orig_w * ratio, orig_h * ratio

                rl_img = RLImage(effective_img_path, width=final_w, height=final_h)
                img_table = Table([[rl_img]], colWidths=[520])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ('PADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(Paragraph("<b>Submitted Visual Inspection Frame:</b>", style_h2))
                story.append(img_table)
                story.append(Spacer(1, 8))
            except Exception:
                pass

        # SECTION 1: Observable Evidence Breakdown
        story.append(Paragraph("1. Observable Evidence & Diagnostic Breakdown", style_h2))
        evidence_data = [
            [Paragraph("<b>Visible Evidence:</b>", style_bold_body), Paragraph(result.visible_evidence, style_body)],
            [Paragraph("<b>Possible Explanation:</b>", style_bold_body), Paragraph(result.possible_explanation, style_body)],
            [Paragraph("<b>Unverified Condition:</b>", style_bold_body), Paragraph(result.unverified_condition, style_body)],
        ]
        ev_table = Table(evidence_data, colWidths=[125, 395])
        ev_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(ev_table)
        story.append(Spacer(1, 6))

        # SECTION 2: Safety Hazards & Risk Analysis
        story.append(Paragraph("2. Safety Hazards & Risk Analysis", style_h2))
        hazard_items = [[Paragraph(f"• {h}", style_body)] for h in result.safety_hazards] or [[Paragraph("No supported hazards identified.", style_body)]]
        haz_table = Table(hazard_items, colWidths=[520])
        haz_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#fcd34d")),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(haz_table)
        story.append(Spacer(1, 6))

        # SECTION 3: Recommended PPE
        story.append(Paragraph("3. Recommended Personal Protective Equipment (PPE)", style_h2))
        ppe_items = [[Paragraph(f"• {p}", style_body)] for p in result.required_ppe] or [[Paragraph("Standard facility-approved safety attire.", style_body)]]
        ppe_table = Table(ppe_items, colWidths=[520])
        ppe_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(ppe_table)
        story.append(Spacer(1, 6))

        # SECTION 4: Standard Operating Procedures (SOP)
        story.append(Paragraph("4. Recommended Standard Operating Procedures (SOP)", style_h2))
        sop_rows = [
            [
                Paragraph("<b>Step</b>", style_bold_body),
                Paragraph("<b>Action Title</b>", style_bold_body),
                Paragraph("<b>Type</b>", style_bold_body),
                Paragraph("<b>Directive & Procedure</b>", style_bold_body),
            ]
        ]
        for s in result.recommended_sops:
            crit = " [CRITICAL]" if s.is_safety_critical else ""
            sop_rows.append([
                Paragraph(str(s.step_number), style_body),
                Paragraph(f"<b>{s.title}</b>{crit}", style_body),
                Paragraph(s.action_type, style_body),
                Paragraph(s.description, style_body),
            ])

        sop_table = Table(sop_rows, colWidths=[30, 130, 70, 290])
        sop_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(sop_table)
        story.append(Spacer(1, 6))

        # SECTION 5: Candidate Replacement Parts
        story.append(Paragraph("5. Candidate Replacement Parts (Compatibility Verification Required)", style_h2))
        if result.replacement_parts:
            parts_items = [[Paragraph(f"• {p}", style_body)] for p in result.replacement_parts]
        else:
            parts_items = [[Paragraph("No replacement part identified from available evidence.", style_body)]]
        parts_table = Table(parts_items, colWidths=[520])
        parts_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(parts_table)
        story.append(Spacer(1, 6))

        # SECTION 6: Referenced Technical Standards (Advisory)
        story.append(Paragraph("6. Applicable Technical Standards (Advisory Technical References)", style_h2))
        std_items = [[Paragraph(f"• {std}", style_body)] for std in result.standards_referenced]
        std_table = Table(std_items, colWidths=[520])
        std_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(std_table)
        story.append(Spacer(1, 6))

        # SECTION 7: Grounded Diagnostic Summary
        story.append(Paragraph("7. Grounded Engineering Narrative", style_h2))
        summary_text = result.llm_reasoning_summary or "No grounded narrative recorded."
        summary_p = Paragraph(summary_text.replace("\n", "<br/>"), style_body)
        sum_table = Table([[summary_p]], colWidths=[520])
        sum_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(sum_table)
        story.append(Spacer(1, 8))

        # SECTION 8: Qualified Technician Review & Sign-Off (Blank)
        story.append(Paragraph("8. Qualified Technician Review & Sign-Off", style_h2))
        signoff_data = [
            [
                Paragraph("<b>Technician Name:</b> ___________________________", style_body),
                Paragraph("<b>License / Certification ID:</b> ___________________________", style_body),
            ],
            [
                Paragraph("<b>Physical Inspection Date:</b> ____________________", style_body),
                Paragraph("<b>Approval Signature:</b> ________________________________", style_body),
            ],
            [
                Paragraph("<b>Sign-Off Status:</b> [  ] Approved for Service &nbsp;&nbsp;&nbsp;&nbsp; [  ] Maintenance Action Required &nbsp;&nbsp;&nbsp;&nbsp; [  ] Decommissioned", style_body),
                Paragraph("<i>(Note: This block is for manual field completion.)</i>", style_body),
            ]
        ]
        sign_table = Table(signoff_data, colWidths=[260, 260])
        sign_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#ffffff")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(sign_table)
        story.append(Spacer(1, 8))

        # SECTION 9: Mandatory Safety Disclaimer
        story.append(Paragraph("9. Mandatory Safety Disclaimer", style_h2))
        disc_table = Table([[Paragraph(f"<b>SAFETY ADVISORY:</b> {result.disclaimer}", style_disclaimer)]], colWidths=[520])
        disc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#f59e0b")),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(disc_table)

        # Build document with NumberedCanvas
        doc.build(story, canvasmaker=NumberedCanvas)
        return buffer.getvalue()

    @classmethod
    def save_pdf(cls, result: DiagnosticResult, output_path: str, image_path: Optional[str] = None) -> str:
        """Saves generated PDF directly to a filesystem path."""
        pdf_bytes = cls.to_pdf(result, image_path=image_path)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        return output_path

    @staticmethod
    def to_markdown(result: DiagnosticResult) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        sla_display = result.dispatch_sla_display

        status_badge = {
            VisualInspectionStatus.CONFIRMED_FAULT: "🔴 **POTENTIAL FAULT DETECTED**",
            VisualInspectionStatus.POTENTIAL_FAULT: "🟠 **POTENTIAL FAULT — VERIFICATION REQUIRED**",
            VisualInspectionStatus.NO_DEFECT_DETECTED: "🔵 **NO VISIBLE DEFECT DETECTED**",
            VisualInspectionStatus.DISAGREEMENT: "⚠️ **VISIBLE DAMAGE OBSERVED — MODEL CLASSIFICATION DISAGREES**",
            VisualInspectionStatus.INSPECTION_FAILED: "⚫ **INSPECTION FAILED**",
        }.get(result.visual_inspection_status, result.visual_inspection_status.value)

        sev_badge = {
            SeverityLevel.CRITICAL: "🚨 `CRITICAL`",
            SeverityLevel.HIGH: "⚠️ `HIGH`",
            SeverityLevel.MEDIUM: "⚡ `MEDIUM`",
            SeverityLevel.LOW: "ℹ️ `LOW`",
            SeverityLevel.UNCERTAIN: "❓ `UNCERTAIN`",
            SeverityLevel.NONE_DETECTED: "✅ `NO VISIBLE DAMAGE IDENTIFIED`",
        }.get(result.overall_severity, f"`{result.overall_severity.value}`")

        weather_notice = " *(Escalated due to ambient rain/moisture)*" if result.weather_escalated else ""
        telemetry_label = result.operational_status.value
        if result.is_telemetry_simulated:
            telemetry_label += " [SIMULATED]"

        md = [
            "# ⚡ AI-ASSISTED VISUAL INSPECTION WORK ORDER — RESEARCH PROTOTYPE",
            f"**Report ID:** `{result.report_id}`  ",
            f"**Station Identifier:** `{result.station_id}`  ",
            f"**Hardware Architecture:** `{result.charger_type}`  ",
            f"**Ambient Weather:** `{result.weather_condition}`  ",
            f"**Generated At:** {date_str}  ",
            f"**Visual Inspection Status:** {status_badge}  ",
            f"**Assigned Severity Level:** {sev_badge}{weather_notice}  ",
            f"**Operational Telemetry:** `{telemetry_label}`  ",
            f"**Maintenance Priority:** `{result.maintenance_priority.value}`  ",
            f"**Dispatch SLA:** **{sla_display}**  ",
            f"**Inspected Component:** `{result.component or 'General'}`  ",
            f"**AI Model Score:** **{result.confidence_score*100:.1f}%**  ",
            "",
        ]

        if result.classification_disagreement:
            md.extend([
                "> ⚠️ **CLASSIFICATION DISAGREEMENT:** Visual observation noted physical defect ('"
                + (result.inspector_visual_finding or "Damage observed")
                + f"'), while model predicted HEALTHY ({result.confidence_score*100:.1f}%). Both observations preserved; manual physical verification required.",
                ""
            ])

        md.extend([
            "---",
            "## 1. Observable Evidence & Diagnostic Breakdown",
            f"- **Visible Evidence:** {result.visible_evidence}",
            f"- **Possible Explanation:** {result.possible_explanation}",
            f"- **Unverified Condition:** {result.unverified_condition}",
            "",
            "## 2. Safety Hazards & Warnings",
        ])

        for h in result.safety_hazards:
            md.append(f"- ⚠️ {h}")

        md.extend([
            "",
            "## 3. Recommended Personal Protective Equipment (PPE)",
        ])
        for p in result.required_ppe:
            md.append(f"- 🛡️ {p}")

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
            md.append("- No replacement part identified from available evidence.")

        md.extend([
            "",
            "## 6. Applicable Technical Standards (Advisory)",
        ])
        for std in result.standards_referenced:
            md.append(f"- 📜 {std}")

        md.extend([
            "",
            "## 7. Grounded Engineering Narrative",
            "```text",
            result.llm_reasoning_summary or "No summary recorded.",
            "```",
            "",
            "## 8. Qualified Technician Review & Sign-Off",
            "- **Technician Name:** ___________________________",
            "- **License / Certification ID:** ___________________________",
            "- **Review Date:** ___________________________",
            "- **Status:** [ ] Approved for Service &nbsp;&nbsp; [ ] Action Required &nbsp;&nbsp; [ ] Decommissioned",
            "",
            "## 9. Mandatory Safety Disclaimer",
            "> ⚠️ **SAFETY DISCLAIMER:** " + result.disclaimer
        ])

        return "\n".join(md)

    @staticmethod
    def to_html(result: DiagnosticResult, image_path: Optional[str] = None) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        sla_display = result.dispatch_sla_display

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
        parts_html = "".join([f"<li>🔧 {p}</li>" for p in result.replacement_parts]) or "<li>No replacement part identified from available evidence</li>"
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
                🚨 WEATHER ESCALATION: Severity elevated to CRITICAL due to rain / wet ambient surface conditions.
            </div>
            """

        disagree_banner = ""
        if result.classification_disagreement:
            disagree_banner = f"""
            <div style="background: #fef2f2; border: 2px solid #ef4444; color: #b91c1c; padding: 10px 14px; border-radius: 6px; margin: 12px 0; font-weight: bold;">
                ⚠️ CLASSIFICATION DISAGREEMENT: Physical visual observation reported defect ('{result.inspector_visual_finding}'), while 2D model predicted HEALTHY ({result.confidence_score*100:.1f}%). Both observations preserved; manual physical verification required.
            </div>
            """

        op_status_label = result.operational_status.value
        if result.is_telemetry_simulated:
            op_status_label += " [SIMULATED]"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Assisted Visual Inspection Report - {result.station_id}</title>
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
        .evidence-grid {{
            display: grid;
            grid-template-columns: 180px 1fr;
            gap: 10px;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            padding: 14px;
            border-radius: 6px;
            margin-top: 10px;
        }}
        .signoff-box {{
            border: 1px solid #94a3b8;
            background: #ffffff;
            padding: 16px;
            border-radius: 6px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <h1 style="margin:0 0 8px 0; color: #0d47a1;">⚡ AI-Assisted Visual Inspection Report</h1>
            <div style="color: #64748b; font-size: 0.95em;">Research Prototype Advisory | Preliminary Optical Screening Only</div>

            <div class="meta-grid">
                <div class="meta-item"><strong>Report ID:</strong> {result.report_id}</div>
                <div class="meta-item"><strong>Station ID:</strong> {result.station_id}</div>
                <div class="meta-item"><strong>Hardware Architecture:</strong> {result.charger_type}</div>
                <div class="meta-item"><strong>Ambient Weather:</strong> {result.weather_condition}</div>
                <div class="meta-item"><strong>Timestamp:</strong> {date_str}</div>
                <div class="meta-item"><strong>Component:</strong> {result.component.upper() if result.component else 'GENERAL'}</div>
                <div class="meta-item"><strong>AI Model Score:</strong> {result.confidence_score*100:.1f}%</div>
            </div>

            <div style="margin-top: 16px;">
                <span class="badge">Visual: {result.visual_inspection_status.value}</span>
                <span class="badge" style="margin-left: 8px;">Severity: {result.overall_severity.value}</span>
                <span class="badge" style="margin-left: 8px; background: #e0f2fe; color: #0369a1; border-color: #7dd3fc;">SLA: {sla_display}</span>
            </div>
            <div style="margin-top: 10px; font-size: 0.9em; color: #555;">
                <strong>Operational Telemetry:</strong> <code>{op_status_label}</code> &nbsp;|&nbsp;
                <strong>Priority:</strong> {result.maintenance_priority.value}
            </div>
        </div>

        {disagree_banner}
        {weather_banner}

        <h2>1. Observable Evidence & Diagnostic Breakdown</h2>
        <div class="evidence-grid">
            <div><strong>Visible Evidence:</strong></div><div>{result.visible_evidence}</div>
            <div><strong>Possible Explanation:</strong></div><div>{result.possible_explanation}</div>
            <div><strong>Unverified Condition:</strong></div><div>{result.unverified_condition}</div>
        </div>

        <h2>2. Safety Hazards & Risk Analysis</h2>
        <div class="hazards-box">
            <ul>{hazards_html}</ul>
        </div>

        <h2>3. Recommended Personal Protective Equipment (PPE)</h2>
        <ul>{ppe_html}</ul>

        <h2>4. Recommended Standard Operating Procedures (SOP)</h2>
        {sops_html}

        <h2>5. Candidate Replacement Parts</h2>
        <ul>{parts_html}</ul>

        <h2>6. Applicable Technical Standards (Advisory)</h2>
        <ul>{stds_html}</ul>

        <h2>7. Grounded Engineering Narrative</h2>
        <div class="summary-pre">{result.llm_reasoning_summary or 'No summary recorded.'}</div>

        <h2>8. Qualified Technician Review & Sign-Off</h2>
        <div class="signoff-box">
            <p><strong>Technician Name:</strong> ___________________________ &nbsp;&nbsp;&nbsp;&nbsp; <strong>License / Cert ID:</strong> ___________________________</p>
            <p><strong>Inspection Date:</strong> ____________________ &nbsp;&nbsp;&nbsp;&nbsp; <strong>Approval Signature:</strong> ________________________________</p>
            <p><strong>Sign-Off Decision:</strong> [  ] Approved for Service &nbsp;&nbsp;&nbsp;&nbsp; [  ] Maintenance Action Required &nbsp;&nbsp;&nbsp;&nbsp; [  ] Decommissioned</p>
            <p style="font-size:0.85em; color:#64748b; margin-top:8px;"><i>(Note: This section is to be completed manually by qualified personnel upon physical site inspection.)</i></p>
        </div>

        <h2>9. Mandatory Safety Disclaimer</h2>
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
