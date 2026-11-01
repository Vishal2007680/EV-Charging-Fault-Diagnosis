"""
Central, Testable Status Assessment Engine for EV Charging Diagnostics.
Supports both:
  1. Object detection bounding boxes (DetectionItem from YOLOv11)
  2. Dual-head classification (CVInferenceResult from MobileNetV3)
Adheres to strict safety engineering guidelines:
  - Never defaults to 'NORMAL' when defects or uncertainties exist.
  - Distinguishes visual inspection status from operational telemetry status.
  - Handles duplicate detections via IoU non-maximum suppression.
  - Explicitly flags low-confidence or ambiguous detections as POTENTIAL FAULT.
  - Escalates coupler faults to CRITICAL under wet/rain weather conditions.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

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
    """
    Suppresses redundant overlapping detections of the same category,
    keeping the detection with the higher confidence score.
    """
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


FAULT_SEVERITY_MAP: Dict[str, Dict[str, Any]] = {
    FaultClass.DAMAGED_CABLE.value: {
        "severity": SeverityLevel.CRITICAL,
        "priority": MaintenancePriority.IMMEDIATE,
        "dispatch_sla": 2,
        "hazard": "Visible cable damage / sheath breach creates potential high-voltage dielectric breakdown and electrocution hazard."
    },
    FaultClass.DAMAGED_CONNECTOR.value: {
        "severity": SeverityLevel.HIGH,
        "priority": MaintenancePriority.URGENT,
        "dispatch_sla": 12,
        "hazard": "Visible connector head or latch damage threatens poor contact mating, mechanical locking failure, or arcing."
    },
    FaultClass.DAMAGED_SCREEN.value: {
        "severity": SeverityLevel.MEDIUM,
        "priority": MaintenancePriority.SCHEDULED,
        "dispatch_sla": 36,
        "hazard": "Cracked display compromises user interface and may breach enclosure IP ingress weather-seal."
    },
    FaultClass.CORROSION_WEAR.value: {
        "severity": SeverityLevel.LOW,
        "priority": MaintenancePriority.ROUTINE,
        "dispatch_sla": 72,
        "hazard": "Surface corrosion or physical wear requiring preventive maintenance."
    }
}


def resolve_operational_status(ctx: ChargerContext) -> OperationalStatus:
    """Derives operational status exclusively from verified OCPP telemetry."""
    if not ctx.has_ocpp_telemetry or not ctx.ocpp_state_raw:
        return OperationalStatus.UNAVAILABLE
    raw = ctx.ocpp_state_raw.lower().strip()
    if "operative" in raw or "available" in raw or "charging" in raw:
        return OperationalStatus.OPERATIVE
    elif "fault" in raw or "inoperative" in raw or "emergency" in raw:
        return OperationalStatus.FAULTED
    elif "restrict" in raw or "derate" in raw or "suspended" in raw:
        return OperationalStatus.RESTRICTED
    return OperationalStatus.UNAVAILABLE


class StatusAssessor:
    """Central status and severity evaluation engine supporting both CV paradigms."""

    CONFIRMED_THRESHOLD = 0.40  # Detections >= 0.40 are treated as supported defects (YOLO)
    UNCERTAIN_THRESHOLD = 0.15  # Detections between 0.15 and 0.40 are flagged for verification (YOLO)
    CONFIDENCE_CONFIRMED_THRESHOLD = 0.65  # Confidence threshold for MobileNet classifier

    @classmethod
    def resolve_operational_status(cls, ctx: ChargerContext) -> OperationalStatus:
        return resolve_operational_status(ctx)

    @classmethod
    def assess(
        cls,
        cv_result: CVInferenceResult,
        context: Optional[ChargerContext] = None
    ) -> Dict[str, Any]:
        """Evaluates MobileNetV3 dual-head CV result (Member 2)."""
        ctx = context or ChargerContext()
        op_status = cls.resolve_operational_status(ctx)

        if cv_result.status != "success":
            err_msg = cv_result.error_message or "Unknown inference or image decoding error."
            return {
                "visual_inspection_status": VisualInspectionStatus.INSPECTION_FAILED,
                "operational_status": op_status,
                "overall_severity": SeverityLevel.UNCERTAIN,
                "maintenance_priority": MaintenancePriority.PENDING_VERIFICATION,
                "confidence_score": 0.0,
                "component": cv_result.component or "unknown",
                "condition": cv_result.condition or "unknown",
                "dispatch_sla_hours": None,
                "weather_escalated": False,
                "safety_hazards": [
                    f"INSPECTION FAILURE: {err_msg}",
                    "Equipment condition cannot be verified from camera feed. On-site manual technician verification required."
                ],
                "required_ppe": ["Standard field technician PPE (safety boots, safety glasses)"],
                "fault_explanations": [f"Image inspection failed: {err_msg}"],
                "root_cause_analysis": "Automated inspection could not be completed. Model or image input error.",
                "recommended_sops": [
                    MaintenanceAction(
                        step_number=1,
                        title="Manual Physical Inspection",
                        description="Dispatch field technician to perform visual and physical examination of the charging station.",
                        is_safety_critical=False,
                        action_type="Inspection"
                    )
                ],
                "replacement_parts": [],
                "status_rationale": f"Inspection failed: {err_msg}. Never default to safe/normal when diagnosis fails."
            }

        component = (cv_result.component or "unknown").lower().strip()
        condition = (cv_result.condition or "unknown").lower().strip()
        cond_conf = cv_result.condition_confidence
        comp_conf = cv_result.component_confidence

        profile = get_fault_profile(component, condition)

        weather_lower = (ctx.weather_condition or "").lower()
        is_wet_weather = any(w in weather_lower for w in ["rain", "wet", "puddle", "water", "storm"])
        is_extreme_heat = any(w in weather_lower for w in ["heat", ">45", "hot", "sun"])

        # Case 2: Broken Condition with Confirmed Confidence (>= 0.65)
        if condition == "broken" and cond_conf >= cls.CONFIDENCE_CONFIRMED_THRESHOLD:
            severity = profile.get("base_severity", SeverityLevel.HIGH)
            priority = profile.get("base_priority", MaintenancePriority.URGENT)
            sla = profile.get("dispatch_sla_hours", 24)
            weather_escalated = False

            hazards = list(profile.get("safety_hazards", []))
            sops = list(profile.get("sops", []))
            ppe = list(profile.get("required_ppe", []))
            parts = list(profile.get("replacement_parts", []))

            if is_wet_weather and component == "plug":
                severity = SeverityLevel.CRITICAL
                priority = MaintenancePriority.IMMEDIATE
                sla = 2
                weather_escalated = True
                hazards.insert(
                    0,
                    "🚨 CRITICAL WEATHER ESCALATION: Active rain / wet surface with a damaged plug creates an immediate, life-threatening arc-flash and ground electrocution hazard!"
                )

            if is_extreme_heat:
                hazards.append("⚠️ THERMAL WARNING: Ambient temperature (>45°C) increases insulation thermal degradation rate.")

            explanation = (
                f"[{component.upper()} DAMAGED] Visual condition: BROKEN (Confidence: {cond_conf*100:.1f}%, Component Conf: {comp_conf*100:.1f}%). "
                f"Observable: {profile['evidence']} "
                f"Hypothesis: {profile['hypothesis']}"
            )

            rationale = (
                f"Confirmed visual defect detected on '{component}' with {cond_conf*100:.1f}% confidence. "
                f"Assigned severity: {severity.value}."
            )
            if weather_escalated:
                rationale += " Escalated to CRITICAL due to wet ambient weather conditions."

            return {
                "visual_inspection_status": VisualInspectionStatus.CONFIRMED_FAULT,
                "operational_status": op_status,
                "overall_severity": severity,
                "maintenance_priority": priority,
                "confidence_score": cond_conf,
                "component": component,
                "condition": condition,
                "dispatch_sla_hours": sla,
                "weather_escalated": weather_escalated,
                "safety_hazards": hazards,
                "required_ppe": ppe,
                "fault_explanations": [explanation],
                "root_cause_analysis": profile["hypothesis"],
                "recommended_sops": sops,
                "replacement_parts": parts,
                "status_rationale": rationale
            }

        # Case 3: Low-Confidence Detection (< 0.65)
        if condition == "broken" and cond_conf < cls.CONFIDENCE_CONFIRMED_THRESHOLD:
            hazards = [
                f"MODERATE CONFIDENCE ANOMALY: Vision model flagged potential damage on '{component}' with {cond_conf*100:.1f}% confidence.",
                "Visual indication is inconclusive. On-site physical technician verification is mandatory before taking unit out of service or approving operation."
            ]
            if is_wet_weather:
                hazards.append("CAUTION: Wet ambient conditions present. Treat unverified anomalies with elevated caution.")

            explanation = (
                f"[{component.upper()} UNVERIFIED ANOMALY] Condition: BROKEN (Confidence: {cond_conf*100:.1f}%). "
                f"Confidence is below confirmed threshold ({cls.CONFIDENCE_CONFIRMED_THRESHOLD*100:.0f}%). "
                f"Initial CV Hint: {cv_result.diagnostic_hint or 'Visual anomaly observed.'}"
            )

            verification_sops = [
                MaintenanceAction(
                    step_number=1,
                    title="On-Site Visual & Tactile Verification",
                    description=f"Inspect the {component} closely for subtle hairline cracks, pin alignment, or sheath tears that triggered the low-confidence classification.",
                    is_safety_critical=False,
                    action_type="Inspection"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Secondary Photo Capture & Diagnostic Re-run",
                    description="Capture a well-lit, close-up photograph from multiple angles and submit for re-inspection.",
                    is_safety_critical=False,
                    action_type="Inspection"
                )
            ]

            return {
                "visual_inspection_status": VisualInspectionStatus.POTENTIAL_FAULT,
                "operational_status": op_status,
                "overall_severity": SeverityLevel.UNCERTAIN,
                "maintenance_priority": MaintenancePriority.PENDING_VERIFICATION,
                "confidence_score": cond_conf,
                "component": component,
                "condition": condition,
                "dispatch_sla_hours": 24,
                "weather_escalated": False,
                "safety_hazards": hazards,
                "required_ppe": ["Standard field technician PPE (safety boots, safety glasses, general work gloves)"],
                "fault_explanations": [explanation],
                "root_cause_analysis": f"Potential visual anomaly on {component} with low AI confidence ({cond_conf*100:.1f}%). Requires technician on-site confirmation.",
                "recommended_sops": verification_sops,
                "replacement_parts": [],
                "status_rationale": f"Vision confidence is moderate ({cond_conf*100:.1f}% < {cls.CONFIDENCE_CONFIRMED_THRESHOLD*100:.0f}%). Visual inspection status set to POTENTIAL FAULT."
            }

        # Case 4: Healthy Condition
        hazards = [
            f"No visible surface defects detected on '{component}' (AI Confidence: {cond_conf*100:.1f}%).",
            "CRITICAL SAFETY LIMITATION: Optical camera inspection cannot detect internal contact resistance, insulation degradation, loose internal busbars, or thermal overheating. Do not treat as proof of electrical safety."
        ]

        explanation = (
            f"[{component.upper()} STRUCTURALLY INTACT] Condition: HEALTHY (Confidence: {cond_conf*100:.1f}%, Component Conf: {comp_conf*100:.1f}%). "
            f"Observable: {profile['evidence']} "
            f"Note: Internal electrical parameters remain unverified by optical camera."
        )

        return {
            "visual_inspection_status": VisualInspectionStatus.NO_DEFECT_DETECTED,
            "operational_status": op_status,
            "overall_severity": SeverityLevel.NONE_DETECTED,
            "maintenance_priority": MaintenancePriority.NONE_REQUIRED,
            "confidence_score": cond_conf,
            "component": component,
            "condition": condition,
            "dispatch_sla_hours": None,
            "weather_escalated": False,
            "safety_hazards": hazards,
            "required_ppe": profile.get("required_ppe", ["Standard field technician PPE"]),
            "fault_explanations": [explanation],
            "root_cause_analysis": f"Component '{component}' appears structurally intact. No surface defect observed.",
            "recommended_sops": profile.get("sops", []),
            "replacement_parts": [],
            "status_rationale": f"Optical screening detected no surface defects on '{component}'. Operational telemetry must be checked independently."
        }

    @classmethod
    def assess_status(
        cls,
        detections: List[DetectionItem],
        context: Optional[ChargerContext] = None,
        inference_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculates status from YOLO object detector bounding boxes."""
        ctx = context or ChargerContext()

        if inference_error:
            return {
                "visual_inspection_status": VisualInspectionStatus.INSPECTION_FAILED,
                "operational_status": OperationalStatus.UNAVAILABLE,
                "overall_severity": SeverityLevel.UNCERTAIN,
                "maintenance_priority": MaintenancePriority.PENDING_VERIFICATION,
                "confidence_score": 0.0,
                "dispatch_sla_hours": None,
                "weather_escalated": False,
                "filtered_detections": [],
                "confirmed_faults": [],
                "uncertain_faults": [],
                "safety_hazards": [f"Inspection failure: {inference_error}. Manual on-site check required."],
                "status_rationale": "Inspection failed due to input or model error. Equipment status cannot be determined."
            }

        deduped = deduplicate_detections(detections, iou_threshold=0.45)

        confirmed_faults: List[DetectionItem] = []
        uncertain_faults: List[DetectionItem] = []
        component_only: List[DetectionItem] = []
        safety_hazards: List[str] = []

        for d in deduped:
            cls_name = d.class_name.lower().strip()
            if cls_name in FAULT_SEVERITY_MAP:
                if d.confidence >= cls.CONFIRMED_THRESHOLD:
                    d.is_uncertain = False
                    confirmed_faults.append(d)
                    safety_hazards.append(FAULT_SEVERITY_MAP[cls_name]["hazard"])
                elif d.confidence >= cls.UNCERTAIN_THRESHOLD:
                    d.is_uncertain = True
                    uncertain_faults.append(d)
                    safety_hazards.append(f"Potential low-confidence anomaly ({cls_name}, {d.confidence:.1%}): On-site verification needed.")
            elif cls_name in ["plug", "cable", "screen", "body"]:
                component_only.append(d)
            else:
                if d.confidence >= cls.UNCERTAIN_THRESHOLD:
                    d.is_uncertain = True
                    uncertain_faults.append(d)
                    safety_hazards.append(f"Unclassified visual anomaly detected ({d.class_name}): Verification needed.")

        is_rain = ctx.weather_condition and any(w in ctx.weather_condition.lower() for w in ["rain", "wet"])
        weather_escalated = False

        if confirmed_faults:
            sev_rank = {
                SeverityLevel.CRITICAL: 4,
                SeverityLevel.HIGH: 3,
                SeverityLevel.MEDIUM: 2,
                SeverityLevel.LOW: 1,
            }
            highest_sev = SeverityLevel.LOW
            lowest_sla = 999
            highest_priority = MaintenancePriority.ROUTINE

            for f in confirmed_faults:
                f_info = FAULT_SEVERITY_MAP.get(f.class_name.lower(), {})
                sev = f_info.get("severity", SeverityLevel.LOW)

                if is_rain and f.class_name.lower() in [FaultClass.DAMAGED_CONNECTOR.value, "damaged_connector"]:
                    sev = SeverityLevel.CRITICAL
                    weather_escalated = True
                    safety_hazards.append("WEATHER ESCALATION: Rain / wet conditions elevate connector damage to an immediate arc and shock hazard.")

                if sev_rank.get(sev, 0) > sev_rank.get(highest_sev, 0):
                    highest_sev = sev
                    highest_priority = MaintenancePriority.IMMEDIATE if sev == SeverityLevel.CRITICAL else f_info.get("priority", MaintenancePriority.URGENT)

                sla = 2 if (weather_escalated and f.class_name.lower() in [FaultClass.DAMAGED_CONNECTOR.value, "damaged_connector"]) else f_info.get("dispatch_sla", 24)
                if sla < lowest_sla:
                    lowest_sla = sla

            max_conf = max(f.confidence for f in confirmed_faults)

            return {
                "visual_inspection_status": VisualInspectionStatus.CONFIRMED_FAULT,
                "operational_status": resolve_operational_status(ctx),
                "overall_severity": highest_sev,
                "maintenance_priority": highest_priority,
                "confidence_score": max_conf,
                "dispatch_sla_hours": lowest_sla if lowest_sla != 999 else 24,
                "weather_escalated": weather_escalated,
                "filtered_detections": deduped,
                "confirmed_faults": confirmed_faults,
                "uncertain_faults": uncertain_faults,
                "safety_hazards": safety_hazards,
                "status_rationale": f"Detected {len(confirmed_faults)} confirmed defect(s). Visual inspection status set to CONFIRMED FAULT."
            }

        if uncertain_faults:
            max_conf = max(u.confidence for u in uncertain_faults)
            return {
                "visual_inspection_status": VisualInspectionStatus.POTENTIAL_FAULT,
                "operational_status": resolve_operational_status(ctx),
                "overall_severity": SeverityLevel.UNCERTAIN,
                "maintenance_priority": MaintenancePriority.PENDING_VERIFICATION,
                "confidence_score": max_conf,
                "dispatch_sla_hours": 48,
                "weather_escalated": False,
                "filtered_detections": deduped,
                "confirmed_faults": [],
                "uncertain_faults": uncertain_faults,
                "safety_hazards": safety_hazards,
                "status_rationale": "Model produced low-confidence or ambiguous detections. On-site verification required before declaring safe."
            }

        comp_summary = f"Located {len(component_only)} component(s): {', '.join([c.class_name for c in component_only])}." if component_only else "No components or defects recognized."

        return {
            "visual_inspection_status": VisualInspectionStatus.NO_DEFECT_DETECTED,
            "operational_status": resolve_operational_status(ctx),
            "overall_severity": SeverityLevel.NONE_DETECTED,
            "maintenance_priority": MaintenancePriority.NONE_REQUIRED,
            "confidence_score": 0.0,
            "dispatch_sla_hours": None,
            "weather_escalated": False,
            "filtered_detections": deduped,
            "confirmed_faults": [],
            "uncertain_faults": [],
            "safety_hazards": [
                "No supported visual defect detected in the current image. NOTE: Visual inspection does not independently confirm internal electrical safety, insulation resistance, or system operation."
            ],
            "status_rationale": f"{comp_summary} No supported visible defect detected. Inspection does not replace qualified testing."
        }
