"""
Central, Testable Status Assessment Engine for EV Charging Diagnostics.
Adheres strictly to evidence-based safety engineering:
  - Formulates explainable severity and maintenance priorities.
  - Distinguishes visual inspection status from operational telemetry status.
  - Populates 3-category observable evidence (visible, hypothesis, unverified).
  - Flags low-confidence predictions (< 0.65) as Low Confidence — Verification Required.
  - Handles visual inspection disagreements when damage is observed but model predicts healthy.
  - Keeps operational status as 'Operational status unavailable — no telemetry received' when disconnected.
  - Clearly identifies simulated telemetry vs verified live connections.
  - Treats dispatch SLA as 'Not assigned' unless specifically configured.
"""

from typing import List, Dict, Any, Tuple, Optional
import uuid
import datetime

from .schema import (
    DetectionItem,
    ChargerContext,
    SeverityLevel,
    VisualInspectionStatus,
    OperationalStatus,
    MaintenancePriority,
    FaultClass,
    CVInferenceResult,
    MaintenanceAction,
)
from .rules import get_fault_profile, FAULT_RULE_DETAILS


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def deduplicate_detections(
    detections: List[DetectionItem],
    iou_threshold: float = 0.50
) -> List[DetectionItem]:
    """Suppresses redundant overlapping detections of the same category."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[DetectionItem] = []

    for det in sorted_dets:
        should_keep = True
        for existing in kept:
            if det.bbox and existing.bbox:
                same_class = (det.class_name.lower() == existing.class_name.lower())
                iou = calculate_iou(det.bbox, existing.bbox)
                if same_class and iou >= iou_threshold:
                    should_keep = False
                    break
        if should_keep:
            kept.append(det)

    return kept


def resolve_operational_status(ctx: ChargerContext) -> OperationalStatus:
    """Derives operational status exclusively from verified telemetry."""
    if not ctx.has_ocpp_telemetry:
        return OperationalStatus.UNAVAILABLE
    if ctx.is_telemetry_simulated:
        return OperationalStatus.SIMULATED
    if not ctx.ocpp_state_raw:
        return OperationalStatus.UNAVAILABLE

    raw = ctx.ocpp_state_raw.lower().strip()
    if "operative" in raw or "available" in raw or "charging" in raw:
        return OperationalStatus.OPERATIVE
    elif "fault" in raw or "inoperative" in raw or "emergency" in raw:
        return OperationalStatus.FAULTED
    elif "restrict" in raw or "derate" in raw or "suspended" in raw:
        return OperationalStatus.RESTRICTED
    return OperationalStatus.UNAVAILABLE


def generate_report_id(station_id: Optional[str]) -> str:
    """Generates unique report ID."""
    now_str = datetime.datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    s = (station_id or "").strip()
    clean_station = "".join(c for c in s if c.isalnum() or c == "-") or "EVCS-UNASSIGNED"
    return f"WO-{clean_station}-{now_str}-{short_uuid}"


class StatusAssessor:
    """Central status and severity evaluation engine."""

    CONFIRMED_THRESHOLD = 0.40
    UNCERTAIN_THRESHOLD = 0.15
    CONFIDENCE_CONFIRMED_THRESHOLD = 0.65

    @classmethod
    def resolve_operational_status(cls, ctx: ChargerContext) -> OperationalStatus:
        return resolve_operational_status(ctx)

    @classmethod
    def assess(
        cls,
        cv_result: CVInferenceResult,
        context: Optional[ChargerContext] = None
    ) -> Dict[str, Any]:
        """Evaluates MobileNetV3 dual-head CV result."""
        ctx = context or ChargerContext()
        op_status = cls.resolve_operational_status(ctx)
        report_id = generate_report_id(ctx.station_id)

        # Failure / Error handling
        if cv_result.status != "success":
            err_msg = cv_result.error_message or "Unknown inference or image decoding error."
            return {
                "report_id": report_id,
                "visual_inspection_status": VisualInspectionStatus.INSPECTION_FAILED,
                "operational_status": op_status,
                "overall_severity": SeverityLevel.UNCERTAIN,
                "maintenance_priority": MaintenancePriority.PENDING_VERIFICATION,
                "confidence_score": 0.0,
                "component": cv_result.component or "unknown",
                "condition": cv_result.condition or "unknown",
                "dispatch_sla_hours": None,
                "weather_escalated": False,
                "status_display_label": "Inspection Failed — Review Required",
                "is_telemetry_simulated": ctx.is_telemetry_simulated,
                "inspector_visual_finding": ctx.inspector_visual_finding,
                "classification_disagreement": False,
                "visible_evidence": f"Image processing failed: {err_msg}",
                "possible_explanation": "Model execution error, unreadable format, or corrupt image file.",
                "unverified_condition": "Equipment condition cannot be verified from camera feed.",
                "safety_hazards": [
                    f"INSPECTION ERROR: {err_msg}",
                    "Equipment condition is unverified. Manual inspection by qualified personnel is required."
                ],
                "required_ppe": ["Task-appropriate PPE as required by on-site facility safety policy."],
                "fault_explanations": [f"Image inspection failed: {err_msg}"],
                "root_cause_analysis": "Automated inspection could not be completed. Manual inspection required.",
                "recommended_sops": [
                    MaintenanceAction(
                        step_number=1,
                        title="Manual Physical Verification",
                        description="Dispatch qualified personnel to perform visual and physical examination of the charging station.",
                        is_safety_critical=False,
                        action_type="Inspection"
                    )
                ],
                "replacement_parts": [],
                "status_rationale": f"Inspection failed: {err_msg}. Never default to safe when diagnosis fails."
            }

        component = (cv_result.component or "unknown").lower().strip()
        condition = (cv_result.condition or "unknown").lower().strip()
        cond_conf = cv_result.condition_confidence
        comp_conf = cv_result.component_confidence

        weather_lower = (ctx.weather_condition or "").lower()
        is_wet_weather = any(w in weather_lower for w in ["rain", "wet", "puddle", "water", "storm"])
        is_extreme_heat = any(w in weather_lower for w in ["heat", ">45", "hot", "sun"])

        # Display Label mapping
        comp_display = {
            "cord": "Cord",
            "plug": "Plug",
            "screen": "Screen",
            "charger": "Charger Enclosure",
        }.get(component, component.capitalize())

        # Check for inspector-reported damage
        inspector_finding = (ctx.inspector_visual_finding or "").lower().strip()
        damage_keywords = ["damage", "broken", "shatter", "crack", "torn", "fray", "exposed", "bent", "defect", "impact"]
        has_inspector_reported_damage = any(k in inspector_finding for k in damage_keywords)

        # -------------------------------------------------------------
        # Case 1: Disagreement — Inspector/Visual Damage Observed, but Model Predicts Healthy
        # -------------------------------------------------------------
        if has_inspector_reported_damage and condition == "healthy":
            broken_profile = get_fault_profile(component, "broken")
            base_sev = broken_profile.get("base_severity", SeverityLevel.HIGH)
            base_prio = broken_profile.get("base_priority", MaintenancePriority.URGENT)

            disagree_hazards = [
                f"MODEL-INSPECTOR DISAGREEMENT: Visual findings report physical defect ('{ctx.inspector_visual_finding}'), while 2D classification model predicted healthy (score: {cond_conf*100:.1f}%).",
                f"POTENTIAL FAILURE HAZARD: Prioritize physical observation over model score until qualified on-site verification occurs.",
                "SAFETY DIRECTIVE: Do not energize or return equipment to service based on automated model score alone."
            ] + list(broken_profile.get("safety_hazards", []))

            explanation = (
                f"[{component.upper()} CLASSIFIER DISAGREEMENT] Visual Observation: '{ctx.inspector_visual_finding}' | "
                f"Model Output: HEALTHY (Score: {cond_conf*100:.1f}%, Component Score: {comp_conf*100:.1f}%). "
                f"Display illumination, glare, or subtle local fracture geometry may cause 2D classification false negative."
            )

            disagree_sops = [
                MaintenanceAction(
                    step_number=1,
                    title="Physical Safety Verification & Access Restriction",
                    description=f"Qualified personnel must physically inspect the {component} and confirm observed damage before energizing.",
                    is_safety_critical=True,
                    action_type="Safety"
                )
            ] + list(broken_profile.get("sops", []))

            return {
                "report_id": report_id,
                "visual_inspection_status": VisualInspectionStatus.DISAGREEMENT,
                "operational_status": op_status,
                "overall_severity": base_sev,
                "maintenance_priority": base_prio,
                "confidence_score": cond_conf,
                "component": component,
                "condition": condition,
                "dispatch_sla_hours": None,
                "weather_escalated": False,
                "status_display_label": "Visible Damage Observed — Model Classification Disagrees; Manual Verification Required",
                "is_telemetry_simulated": ctx.is_telemetry_simulated,
                "inspector_visual_finding": ctx.inspector_visual_finding,
                "classification_disagreement": True,
                "visible_evidence": f"Visible damage observed or reported: '{ctx.inspector_visual_finding}'. Model condition score: {cond_conf*100:.1f}% HEALTHY.",
                "possible_explanation": "Illuminated display graphics, reflections, or localized fracture patterns may have caused classifier false negative. Preserving both findings.",
                "unverified_condition": "Electrical safety, dielectric strength, and functional condition cannot be confirmed; manual physical verification required.",
                "safety_hazards": disagree_hazards,
                "required_ppe": broken_profile.get("required_ppe", ["Task-appropriate PPE per site safety evaluation."]),
                "fault_explanations": [explanation],
                "root_cause_analysis": "Discrepancy between optical image classifier and physical visual inspection. Priority given to physical safety.",
                "recommended_sops": disagree_sops,
                "replacement_parts": broken_profile.get("replacement_parts", []),
                "status_rationale": f"Visible Damage Observed — Model Classification Disagrees; Manual Verification Required (Model: {cond_conf*100:.1f}% healthy)."
            }

        # -------------------------------------------------------------
        # Case 2: Broken Condition with Confirmed Confidence (>= 0.65)
        # -------------------------------------------------------------
        profile = get_fault_profile(component, condition)

        if condition == "broken" and cond_conf >= cls.CONFIDENCE_CONFIRMED_THRESHOLD:
            severity = profile.get("base_severity", SeverityLevel.HIGH)
            priority = profile.get("base_priority", MaintenancePriority.URGENT)
            sla = profile.get("dispatch_sla_hours", None)
            weather_escalated = False

            hazards = list(profile.get("safety_hazards", []))
            sops = list(profile.get("sops", []))
            ppe = list(profile.get("required_ppe", []))
            parts = list(profile.get("replacement_parts", []))

            # Dynamic weather escalation
            if is_wet_weather and component in ["plug", "cord"]:
                severity = SeverityLevel.CRITICAL
                priority = MaintenancePriority.IMMEDIATE
                sla = 2
                weather_escalated = True
                hazards.insert(
                    0,
                    "WEATHER ALERT: Active rain / wet surface conditions present with suspected high-voltage component damage. Elevates dielectric breakdown risk."
                )

            if is_extreme_heat:
                hazards.append("THERMAL CAUTION: High ambient temperature (>45°C) increases insulation thermal stress.")

            status_label = f"Potential {comp_display} Damage Detected"

            explanation = (
                f"[{component.upper()} SUSPECTED DAMAGE] Condition: BROKEN (Model Score: {cond_conf*100:.1f}%, Component Score: {comp_conf*100:.1f}%). "
                f"Visible: {profile['visible_evidence']} "
                f"Hypothesis: {profile['possible_explanation']}"
            )

            rationale = (
                f"Potential visual defect identified on '{component}' with model score {cond_conf*100:.1f}%. "
                f"Assigned severity: {severity.value}."
            )
            if weather_escalated:
                rationale += " Escalated to CRITICAL due to wet ambient weather conditions."

            return {
                "report_id": report_id,
                "visual_inspection_status": VisualInspectionStatus.CONFIRMED_FAULT,
                "operational_status": op_status,
                "overall_severity": severity,
                "maintenance_priority": priority,
                "confidence_score": cond_conf,
                "component": component,
                "condition": condition,
                "dispatch_sla_hours": sla,
                "weather_escalated": weather_escalated,
                "status_display_label": status_label,
                "is_telemetry_simulated": ctx.is_telemetry_simulated,
                "inspector_visual_finding": ctx.inspector_visual_finding,
                "classification_disagreement": False,
                "visible_evidence": profile.get("visible_evidence", "Visible surface damage is suspected."),
                "possible_explanation": profile.get("possible_explanation", "Possible mechanical wear or impact."),
                "unverified_condition": profile.get("unverified_condition", "Internal electrical integrity unverified."),
                "safety_hazards": hazards,
                "required_ppe": ppe,
                "fault_explanations": [explanation],
                "root_cause_analysis": profile.get("possible_explanation", "Pending physical inspection."),
                "recommended_sops": sops,
                "replacement_parts": parts,
                "status_rationale": rationale
            }

        # -------------------------------------------------------------
        # Case 3: Broken Condition with Low-Confidence (< 0.65)
        # -------------------------------------------------------------
        if condition == "broken" and cond_conf < cls.CONFIDENCE_CONFIRMED_THRESHOLD:
            hazards = [
                f"AMBIGUOUS / LOW-CONFIDENCE OBSERVATION: Vision model scored potential damage on '{component}' at {cond_conf*100:.1f}%.",
                "Visual evidence is inconclusive. On-site physical verification by a qualified technician is required."
            ]
            if is_wet_weather:
                hazards.append("CAUTION: Wet ambient conditions present. Treat unverified anomalies with elevated caution.")

            explanation = (
                f"[{component.upper()} UNVERIFIED OBSERVATION] Condition: BROKEN (Model Score: {cond_conf*100:.1f}%). "
                f"Confidence is below confirmed screening threshold ({cls.CONFIDENCE_CONFIRMED_THRESHOLD*100:.0f}%). "
                f"Requires manual on-site physical review."
            )

            verification_sops = [
                MaintenanceAction(
                    step_number=1,
                    title="On-Site Visual & Mechanical Review",
                    description=f"Qualified personnel inspect the {component} closely for subtle wear, pin distortion, or surface tears that triggered the low-confidence classification.",
                    is_safety_critical=False,
                    action_type="Inspection"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Secondary Image Capture",
                    description="If ambiguity remains, capture a clear, well-lit photograph from multiple angles for re-assessment.",
                    is_safety_critical=False,
                    action_type="Inspection"
                )
            ]

            return {
                "report_id": report_id,
                "visual_inspection_status": VisualInspectionStatus.POTENTIAL_FAULT,
                "operational_status": op_status,
                "overall_severity": SeverityLevel.UNCERTAIN,
                "maintenance_priority": MaintenancePriority.PENDING_VERIFICATION,
                "confidence_score": cond_conf,
                "component": component,
                "condition": condition,
                "dispatch_sla_hours": None,
                "weather_escalated": False,
                "status_display_label": f"Potential {comp_display} Damage — Review Required",
                "is_telemetry_simulated": ctx.is_telemetry_simulated,
                "inspector_visual_finding": ctx.inspector_visual_finding,
                "classification_disagreement": False,
                "visible_evidence": f"Ambiguous visual indication on {component} (Score: {cond_conf*100:.1f}%).",
                "possible_explanation": "Model feature response is ambiguous; may represent lighting reflections or minor surface marks.",
                "unverified_condition": "Actual physical condition and electrical safety remain unverified.",
                "safety_hazards": hazards,
                "required_ppe": ["Standard task-appropriate safety attire per facility policy."],
                "fault_explanations": [explanation],
                "root_cause_analysis": f"Potential visual anomaly on {component} with low AI score ({cond_conf*100:.1f}%). Requires on-site confirmation.",
                "recommended_sops": verification_sops,
                "replacement_parts": [],
                "status_rationale": f"Vision score is moderate ({cond_conf*100:.1f}% < {cls.CONFIDENCE_CONFIRMED_THRESHOLD*100:.0f}%). Status set to POTENTIAL FAULT — VERIFICATION REQUIRED."
            }

        # -------------------------------------------------------------
        # Case 4: Healthy Condition with Low Confidence (< 0.65)
        # -------------------------------------------------------------
        if condition == "healthy" and cond_conf < cls.CONFIDENCE_CONFIRMED_THRESHOLD:
            hazards = [
                f"LOW-CONFIDENCE CLASSIFICATION: Vision model scored '{component}' as healthy at {cond_conf*100:.1f}%, which is below the verified decision threshold ({cls.CONFIDENCE_CONFIRMED_THRESHOLD*100:.0f}%).",
                "SAFETY WARNING: Do not assume equipment is defect-free solely from a low-confidence visual prediction. Subtle cracks or damaged elements may be present."
            ]

            explanation = (
                f"[{component.upper()} LOW CONFIDENCE] Condition: HEALTHY (Model Score: {cond_conf*100:.1f}%, Component Score: {comp_conf*100:.1f}%). "
                f"Model score is below screening threshold ({cls.CONFIDENCE_CONFIRMED_THRESHOLD*100:.0f}%). "
                f"Near decision boundary — inconclusive visual screening."
            )

            verification_sops = [
                MaintenanceAction(
                    step_number=1,
                    title="Manual Physical Verification",
                    description=f"Qualified personnel inspect the {component} on-site for subtle hairline cracks, connector wear, or pin distortion before declaring normal operation.",
                    is_safety_critical=True,
                    action_type="Inspection"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Secondary Angle Capture",
                    description="Capture multi-angle imagery under balanced lighting to confirm equipment condition.",
                    is_safety_critical=False,
                    action_type="Inspection"
                )
            ]

            return {
                "report_id": report_id,
                "visual_inspection_status": VisualInspectionStatus.POTENTIAL_FAULT,
                "operational_status": op_status,
                "overall_severity": SeverityLevel.UNCERTAIN,
                "maintenance_priority": MaintenancePriority.PENDING_VERIFICATION,
                "confidence_score": cond_conf,
                "component": component,
                "condition": condition,
                "dispatch_sla_hours": None,
                "weather_escalated": False,
                "status_display_label": "Low Confidence — Verification Required",
                "is_telemetry_simulated": ctx.is_telemetry_simulated,
                "inspector_visual_finding": ctx.inspector_visual_finding,
                "classification_disagreement": False,
                "visible_evidence": f"Inconclusive visual screening on '{component}' (Healthy Prediction Score: {cond_conf*100:.1f}%).",
                "possible_explanation": "Visual patterns lie close to the decision boundary; subtle hairline cracks, surface tears, or internal damage cannot be ruled out by camera inspection.",
                "unverified_condition": "Equipment cannot be verified as healthy from optical screening alone. Manual on-site inspection by qualified personnel is required.",
                "safety_hazards": hazards,
                "required_ppe": ["Standard facility-approved PPE."],
                "fault_explanations": [explanation],
                "root_cause_analysis": f"Model predicted healthy but with low confidence ({cond_conf*100:.1f}% < {cls.CONFIDENCE_CONFIRMED_THRESHOLD*100:.0f}%). Inconclusive visual screening.",
                "recommended_sops": verification_sops,
                "replacement_parts": [],
                "status_rationale": f"Model predicted healthy with low confidence ({cond_conf*100:.1f}% < {cls.CONFIDENCE_CONFIRMED_THRESHOLD*100:.0f}%). Status set to Low Confidence — Verification Required; optical screening cannot confirm healthy status."
            }

        # -------------------------------------------------------------
        # Case 4b: Screen Specific Rule — High Confidence Healthy Prediction
        # Display illumination, glare, or subtle crack geometry can lead to false negative.
        # AI visual classification is not sufficient to certify a screen as healthy.
        # -------------------------------------------------------------
        if component == "screen" and condition == "healthy" and cond_conf >= cls.CONFIDENCE_CONFIRMED_THRESHOLD:
            screen_profile = get_fault_profile("screen", "broken")
            screen_hazards = [
                f"SCREEN INSPECTION REQUIRED: Vision model predicted screen as HEALTHY (AI Score: {cond_conf*100:.1f}%, Component Score: {comp_conf*100:.1f}%).",
                "SAFETY & USABILITY DIRECTIVE: Visual AI models cannot reliably certify screen integrity. Display illumination, backlighting, reflection, or viewing angle can obscure hairline fractures or shattered glass.",
                "Electrical safety, ingress protection (IP seal integrity), and touchscreen functional safety cannot be confirmed from camera imagery alone; qualified on-site physical inspection is required."
            ]

            explanation = (
                f"[SCREEN INSPECTION REQUIRED] Condition: HEALTHY (Model Score: {cond_conf*100:.1f}%, Component Score: {comp_conf*100:.1f}%). "
                f"Screen Inspection Required — AI Result Not Sufficient to Confirm Condition. "
                f"Illuminated display graphics or reflection patterns may conceal mechanical fractures or shattered glass. Manual verification required."
            )

            screen_sops = [
                MaintenanceAction(
                    step_number=1,
                    title="Physical Display & Enclosure Verification",
                    description="Qualified personnel must physically inspect the display surface under oblique lighting to check for glass fractures, seal compromise, or impact damage.",
                    is_safety_critical=True,
                    action_type="Inspection"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Weather-Seal and Moisture Ingress Check",
                    description="Inspect perimeter gasket and bezel sealing around the display panel to ensure moisture or rain cannot enter internal electrical enclosures.",
                    is_safety_critical=False,
                    action_type="Inspection"
                ),
                MaintenanceAction(
                    step_number=3,
                    title="Touch Interface & Emergency UI Test",
                    description="Verify touchscreen responsiveness, emergency stop prompts, and legible visual output across all operational menus.",
                    is_safety_critical=False,
                    action_type="Testing"
                )
            ]

            return {
                "report_id": report_id,
                "visual_inspection_status": VisualInspectionStatus.POTENTIAL_FAULT,
                "operational_status": op_status,
                "overall_severity": SeverityLevel.UNCERTAIN,
                "maintenance_priority": MaintenancePriority.PENDING_VERIFICATION,
                "confidence_score": cond_conf,
                "component": component,
                "condition": condition,
                "dispatch_sla_hours": None,
                "weather_escalated": False,
                "status_display_label": "Screen Inspection Required — AI Result Not Sufficient to Confirm Condition",
                "is_telemetry_simulated": ctx.is_telemetry_simulated,
                "inspector_visual_finding": ctx.inspector_visual_finding,
                "classification_disagreement": False,
                "visible_evidence": f"Screen predicted HEALTHY (Model Score: {cond_conf*100:.1f}%). Illuminated display graphics or reflections may conceal fractures.",
                "possible_explanation": "Screen Inspection Required — 2D visual classification cannot distinguish between an intact illuminated screen and a fractured illuminated screen with certainty. Requires qualified physical inspection.",
                "unverified_condition": "Glass fracture state, enclosure weatherproofing, touch sensor response, and internal electronic condition remain unverified.",
                "safety_hazards": screen_hazards,
                "required_ppe": ["Task-appropriate PPE including safety glasses and cut-resistant gloves if handling glass."],
                "fault_explanations": [explanation],
                "root_cause_analysis": "Screen classified as healthy by visual AI. Screen safety policy mandates manual verification before clearing.",
                "recommended_sops": screen_sops,
                "replacement_parts": screen_profile.get("replacement_parts", []),
                "status_rationale": "Screen Inspection Required — AI Result Not Sufficient to Confirm Condition. Visual AI cannot confirm screen integrity."
            }

        # -------------------------------------------------------------
        # Case 5: Truly Healthy Condition (cond_conf >= 0.65, no reported defects)
        # -------------------------------------------------------------
        hazards = [
            f"No visible surface defects detected on '{component}' in this image (AI Score: {cond_conf*100:.1f}%).",
            "SAFETY WARNING: Electrical safety and internal functionality have not been verified by visual inspection. Internal insulation resistance, contact heating, and grounding continuity cannot be determined from optical imagery."
        ]

        explanation = (
            f"[{component.upper()} NO VISIBLE DEFECT] Condition: HEALTHY (Model Score: {cond_conf*100:.1f}%, Component Score: {comp_conf*100:.1f}%). "
            f"Visible: {profile.get('visible_evidence', 'No visible damage identified.')} "
            f"Note: Internal electrical parameters and mechanical wear remain unverified by optical camera."
        )

        return {
            "report_id": report_id,
            "visual_inspection_status": VisualInspectionStatus.NO_DEFECT_DETECTED,
            "operational_status": op_status,
            "overall_severity": SeverityLevel.NONE_DETECTED,
            "maintenance_priority": MaintenancePriority.ROUTINE,
            "confidence_score": cond_conf,
            "component": component,
            "condition": condition,
            "dispatch_sla_hours": None,
            "weather_escalated": False,
            "status_display_label": "No Visible Defect Detected",
            "is_telemetry_simulated": ctx.is_telemetry_simulated,
            "inspector_visual_finding": ctx.inspector_visual_finding,
            "classification_disagreement": False,
            "visible_evidence": profile.get("visible_evidence", "No visible surface damage identified in this view."),
            "possible_explanation": profile.get("possible_explanation", "Component appears structurally intact from available imagery."),
            "unverified_condition": profile.get("unverified_condition", "Internal electrical integrity and operational status remain unverified."),
            "safety_hazards": hazards,
            "required_ppe": profile.get("required_ppe", ["Standard facility-approved PPE."]),
            "fault_explanations": [explanation],
            "root_cause_analysis": f"Component '{component}' shows no visible surface defects in provided image. Routine maintenance remains subject to manufacturer schedule.",
            "recommended_sops": profile.get("sops", []),
            "replacement_parts": [],
            "status_rationale": f"Optical screening identified no surface defects on '{component}'. Operational telemetry must be verified independently."
        }

    @classmethod
    def assess_status(
        cls,
        detections: List[DetectionItem],
        context: Optional[ChargerContext] = None,
        inference_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculates status from YOLO object detector bounding boxes (backward compatibility)."""
        ctx = context or ChargerContext()
        report_id = generate_report_id(ctx.station_id)

        if inference_error:
            return {
                "report_id": report_id,
                "visual_inspection_status": VisualInspectionStatus.INSPECTION_FAILED,
                "operational_status": OperationalStatus.UNAVAILABLE,
                "overall_severity": SeverityLevel.UNCERTAIN,
                "maintenance_priority": MaintenancePriority.PENDING_VERIFICATION,
                "confidence_score": 0.0,
                "dispatch_sla_hours": None,
                "weather_escalated": False,
                "status_display_label": "Inspection Failed — Input Error",
                "is_telemetry_simulated": ctx.is_telemetry_simulated,
                "inspector_visual_finding": ctx.inspector_visual_finding,
                "classification_disagreement": False,
                "filtered_detections": [],
                "confirmed_faults": [],
                "visible_evidence": f"Inference pipeline error: {inference_error}",
                "possible_explanation": "Model inference error.",
                "unverified_condition": "Station condition unverified.",
                "safety_hazards": [f"PIPELINE ERROR: {inference_error}"],
                "required_ppe": ["Task-appropriate PPE as required by on-site safety policy."],
                "fault_explanations": [f"Model error: {inference_error}"],
                "recommended_sops": [
                    MaintenanceAction(
                        step_number=1,
                        title="Manual Inspection Required",
                        description="Dispatch field technician for manual evaluation.",
                        is_safety_critical=False,
                        action_type="Inspection"
                    )
                ],
                "replacement_parts": [],
            }

        filtered = deduplicate_detections(detections)
        confirmed = [d for d in filtered if d.confidence >= cls.CONFIRMED_THRESHOLD]
        uncertain = [d for d in filtered if cls.UNCERTAIN_THRESHOLD <= d.confidence < cls.CONFIRMED_THRESHOLD]

        weather_lower = (ctx.weather_condition or "").lower()
        is_wet = any(w in weather_lower for w in ["rain", "wet", "water", "storm"])

        if confirmed:
            top_det = max(confirmed, key=lambda d: d.confidence)
            cls_name = top_det.class_name.lower()
            details = FAULT_RULE_DETAILS.get(cls_name, FAULT_RULE_DETAILS["default"])

            sev = details["severity"]
            prio = details["priority"]
            sla = details["sla_hours"]
            weather_escalated = False

            if is_wet and cls_name in ["damaged_connector", "damaged_cable"]:
                sev = SeverityLevel.CRITICAL
                prio = MaintenancePriority.IMMEDIATE
                sla = 2
                weather_escalated = True

            return {
                "report_id": report_id,
                "visual_inspection_status": VisualInspectionStatus.CONFIRMED_FAULT,
                "operational_status": cls.resolve_operational_status(ctx),
                "overall_severity": sev,
                "maintenance_priority": prio,
                "confidence_score": top_det.confidence,
                "dispatch_sla_hours": sla,
                "weather_escalated": weather_escalated,
                "status_display_label": f"Potential {cls_name.replace('_', ' ').title()} Detected",
                "is_telemetry_simulated": ctx.is_telemetry_simulated,
                "inspector_visual_finding": ctx.inspector_visual_finding,
                "classification_disagreement": False,
                "filtered_detections": filtered,
                "confirmed_faults": confirmed,
                "visible_evidence": f"Object detection localized {cls_name} (Confidence: {top_det.confidence*100:.1f}%).",
                "possible_explanation": "Defect detected by localized object bounding box.",
                "unverified_condition": "Internal electrical and telemetry condition unverified.",
                "safety_hazards": list(details["hazards"]),
                "required_ppe": list(details["ppe"]),
                "fault_explanations": [f"Detected {cls_name} at confidence {top_det.confidence*100:.1f}%"],
                "recommended_sops": list(details["sops"]),
                "replacement_parts": list(details["parts"]),
            }
        elif uncertain:
            top_det = max(uncertain, key=lambda d: d.confidence)
            cls_name = top_det.class_name.lower()
            details = FAULT_RULE_DETAILS.get(cls_name, FAULT_RULE_DETAILS["default"])

            return {
                "report_id": report_id,
                "visual_inspection_status": VisualInspectionStatus.POTENTIAL_FAULT,
                "operational_status": cls.resolve_operational_status(ctx),
                "overall_severity": SeverityLevel.UNCERTAIN,
                "maintenance_priority": MaintenancePriority.PENDING_VERIFICATION,
                "confidence_score": top_det.confidence,
                "dispatch_sla_hours": None,
                "weather_escalated": False,
                "status_display_label": f"Potential {cls_name.replace('_', ' ').title()} — Review Required",
                "is_telemetry_simulated": ctx.is_telemetry_simulated,
                "inspector_visual_finding": ctx.inspector_visual_finding,
                "classification_disagreement": False,
                "filtered_detections": filtered,
                "confirmed_faults": [],
                "visible_evidence": f"Ambiguous object detection for {cls_name} (Confidence: {top_det.confidence*100:.1f}%).",
                "possible_explanation": "Low confidence detection near decision threshold.",
                "unverified_condition": "Physical condition requires human verification.",
                "safety_hazards": ["Uncertain detection: manual verification required."],
                "required_ppe": ["Task-appropriate PPE."],
                "fault_explanations": [f"Uncertain {cls_name} detected."],
                "recommended_sops": [
                    MaintenanceAction(
                        step_number=1,
                        title="Manual Verification",
                        description=f"Inspect suspected {cls_name}.",
                        is_safety_critical=False,
                        action_type="Inspection"
                    )
                ],
                "replacement_parts": [],
            }
        else:
            return {
                "report_id": report_id,
                "visual_inspection_status": VisualInspectionStatus.NO_DEFECT_DETECTED,
                "operational_status": cls.resolve_operational_status(ctx),
                "overall_severity": SeverityLevel.NONE_DETECTED,
                "maintenance_priority": MaintenancePriority.ROUTINE,
                "confidence_score": 0.0,
                "dispatch_sla_hours": None,
                "weather_escalated": False,
                "status_display_label": "No Visible Defect Detected",
                "is_telemetry_simulated": ctx.is_telemetry_simulated,
                "inspector_visual_finding": ctx.inspector_visual_finding,
                "classification_disagreement": False,
                "filtered_detections": filtered,
                "confirmed_faults": [],
                "visible_evidence": "No defect bounding boxes detected.",
                "possible_explanation": "No visual defect identified by object detector.",
                "unverified_condition": "Internal electrical and operational status unverified.",
                "safety_hazards": ["Visual inspection only. Electrical safety not verified."],
                "required_ppe": ["Standard safety attire."],
                "fault_explanations": ["No visual defect detected."],
                "recommended_sops": [
                    MaintenanceAction(
                        step_number=1,
                        title="Routine Inspection",
                        description="Follow standard routine maintenance intervals.",
                        is_safety_critical=False,
                        action_type="Inspection"
                    )
                ],
                "replacement_parts": [],
            }
