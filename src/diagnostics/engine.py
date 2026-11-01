"""
Unified Diagnostic Engine for EV Charging Stations.
Coordinates:
  1. Direct ingestion of Member 2's Computer Vision predictions (predict_fault).
  2. Ingestion of Member 1 / YOLO object detections (DetectionItem).
  3. Centralized safety-status assessment & international standard rules (IEC 61851-23 / NFPA 70E).
  4. Grounded LLM reasoning & interactive technician assistant.
  5. Work order generation and export.
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional, List
from PIL import Image
import numpy as np

from .schema import (
    DiagnosticInput,
    DetectionItem,
    CVInferenceResult,
    ChargerContext,
    DiagnosticResult,
    VisualInspectionStatus,
    OperationalStatus,
    SeverityLevel,
    MaintenancePriority,
)
from .status_assessor import StatusAssessor
from .rules import RuleBasedDiagnosticEngine
from .llm_assistant import LLMDiagnosticAssistant

# Import Member 2's inference function
try:
    from infer import predict_fault
except ImportError:
    try:
        from src.infer import predict_fault
    except ImportError:
        predict_fault = None


class EVDiagnosticEngine:
    """
    Main Diagnostic Engine coordinating CV inference ingestion, safety rules,
    and grounded LLM reasoning.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.assessor = StatusAssessor()
        self.rule_engine = RuleBasedDiagnosticEngine()
        self.llm_assistant = LLMDiagnosticAssistant(api_key=api_key)

    def diagnose(self, input_data: Any) -> DiagnosticResult:
        """
        Unified diagnosis entry point taking DiagnosticInput, CVInferenceResult, or dict.
        Handles both YOLO detections and MobileNet CV results.
        """
        if isinstance(input_data, (CVInferenceResult, dict)):
            return self.diagnose_from_cv(input_data)

        context = getattr(input_data, "context", None) or ChargerContext(station_id=getattr(input_data, "station_id", None))
        st_id = context.station_id or getattr(input_data, "station_id", None) or "Not provided"
        ch_type = context.charger_type or "Not provided"
        w_cond = context.weather_condition or "Not provided"

        # Path A: MobileNet CV Result provided
        if input_data.cv_result is not None:
            return self.diagnose_from_cv(input_data.cv_result, context=context, station_id=st_id)

        # Path B: Image path provided
        if input_data.image_path is not None and not input_data.detections:
            return self.diagnose_image(input_data.image_path, context=context, station_id=st_id)

        # Path C: YOLO Detection items provided
        eval_data = self.rule_engine.evaluate(
            detections=input_data.detections,
            context=context,
            inference_error=input_data.inference_error
        )

        llm_summary = self.llm_assistant.generate_synthesis(
            station_id=st_id,
            visual_status=eval_data["visual_inspection_status"],
            operational_status=eval_data["operational_status"],
            severity=eval_data["overall_severity"],
            component=input_data.detections[0].class_name if input_data.detections else "General",
            condition="broken" if eval_data.get("confirmed_faults") else "healthy",
            confidence=eval_data["confidence_score"],
            hazards=eval_data["safety_hazards"],
            explanations=eval_data["fault_explanations"],
            sops=eval_data["recommended_sops"],
            charger_type=ch_type,
            weather=w_cond,
            is_telemetry_simulated=eval_data.get("is_telemetry_simulated", context.is_telemetry_simulated),
            inspector_visual_finding=eval_data.get("inspector_visual_finding", context.inspector_visual_finding),
            classification_disagreement=eval_data.get("classification_disagreement", False)
        )

        root_cause = (
            eval_data.get("status_rationale")
            or "Pending on-site qualified technician verification."
        )

        return DiagnosticResult(
            report_id=eval_data.get("report_id", "WO-EVCS-PRELIM"),
            station_id=st_id,
            charger_type=ch_type,
            weather_condition=w_cond,
            visual_inspection_status=eval_data["visual_inspection_status"],
            operational_status=eval_data["operational_status"],
            overall_severity=eval_data["overall_severity"],
            maintenance_priority=eval_data["maintenance_priority"],
            confidence_score=eval_data["confidence_score"],
            component=input_data.detections[0].class_name if input_data.detections else None,
            condition="broken" if eval_data.get("confirmed_faults") else "healthy",
            dispatch_sla_hours=eval_data["dispatch_sla_hours"],
            weather_escalated=eval_data.get("weather_escalated", False),
            status_display_label=eval_data.get("status_display_label", "Potential Fault Detected"),
            is_telemetry_simulated=eval_data.get("is_telemetry_simulated", context.is_telemetry_simulated),
            inspector_visual_finding=eval_data.get("inspector_visual_finding", context.inspector_visual_finding),
            classification_disagreement=eval_data.get("classification_disagreement", False),
            visible_evidence=eval_data.get("visible_evidence", "Visible anomaly observed."),
            possible_explanation=eval_data.get("possible_explanation", "Hypothesis pending physical inspection."),
            unverified_condition=eval_data.get("unverified_condition", "Internal electrical integrity unverified."),
            safety_hazards=eval_data["safety_hazards"],
            required_ppe=eval_data["required_ppe"],
            fault_explanations=eval_data["fault_explanations"],
            root_cause_analysis=root_cause,
            recommended_sops=eval_data["recommended_sops"],
            replacement_parts=eval_data["replacement_parts"],
            llm_reasoning_summary=llm_summary,
            image_path=input_data.image_path,
        )

    def diagnose_from_cv(
        self,
        cv_data: Union[CVInferenceResult, Dict[str, Any]],
        context: Optional[ChargerContext] = None,
        station_id: Optional[str] = None
    ) -> DiagnosticResult:
        """
        Executes diagnostic evaluation from a structured CV inference result (Member 2).
        """
        ctx = context or ChargerContext(station_id=station_id)
        if station_id and not ctx.station_id:
            ctx.station_id = station_id

        if isinstance(cv_data, dict):
            cv_obj = CVInferenceResult(
                status=cv_data.get("status", "error"),
                condition=cv_data.get("condition"),
                condition_confidence=float(cv_data.get("condition_confidence", 0.0)),
                component=cv_data.get("component"),
                component_confidence=float(cv_data.get("component_confidence", 0.0)),
                is_fault_detected=cv_data.get("is_fault_detected"),
                visual_fault_localized=cv_data.get("visual_fault_localized", False),
                bounding_box=cv_data.get("bounding_box"),
                execution_time_ms=cv_data.get("execution_time_ms"),
                model_version=cv_data.get("model_version"),
                diagnostic_hint=cv_data.get("diagnostic_hint"),
                image_source=cv_data.get("image_source"),
                probabilities=cv_data.get("probabilities"),
                error_message=cv_data.get("error_message"),
            )
        else:
            cv_obj = cv_data

        assessment = self.assessor.assess(cv_result=cv_obj, context=ctx)

        st_id = ctx.station_id or "Not provided"
        ch_type = ctx.charger_type or "Not provided"
        w_cond = ctx.weather_condition or "Not provided"

        llm_summary = self.llm_assistant.generate_synthesis(
            station_id=st_id,
            visual_status=assessment["visual_inspection_status"],
            operational_status=assessment["operational_status"],
            severity=assessment["overall_severity"],
            component=assessment["component"],
            condition=assessment["condition"],
            confidence=assessment["confidence_score"],
            hazards=assessment["safety_hazards"],
            explanations=assessment["fault_explanations"],
            sops=assessment["recommended_sops"],
            charger_type=ch_type,
            weather=w_cond,
            is_telemetry_simulated=assessment.get("is_telemetry_simulated", ctx.is_telemetry_simulated),
            inspector_visual_finding=assessment.get("inspector_visual_finding", ctx.inspector_visual_finding),
            classification_disagreement=assessment.get("classification_disagreement", False)
        )

        image_source = cv_obj.image_source if cv_obj.image_source else None

        return DiagnosticResult(
            report_id=assessment.get("report_id", "WO-EVCS-PRELIM"),
            station_id=st_id,
            charger_type=ch_type,
            weather_condition=w_cond,
            visual_inspection_status=assessment["visual_inspection_status"],
            operational_status=assessment["operational_status"],
            overall_severity=assessment["overall_severity"],
            maintenance_priority=assessment["maintenance_priority"],
            confidence_score=assessment["confidence_score"],
            component=assessment["component"],
            condition=assessment["condition"],
            dispatch_sla_hours=assessment["dispatch_sla_hours"],
            weather_escalated=assessment["weather_escalated"],
            status_display_label=assessment.get("status_display_label", "Potential Fault Detected"),
            is_telemetry_simulated=assessment.get("is_telemetry_simulated", ctx.is_telemetry_simulated),
            inspector_visual_finding=assessment.get("inspector_visual_finding", ctx.inspector_visual_finding),
            classification_disagreement=assessment.get("classification_disagreement", False),
            visible_evidence=assessment.get("visible_evidence", "Visible condition observed."),
            possible_explanation=assessment.get("possible_explanation", "Hypothesis pending physical inspection."),
            unverified_condition=assessment.get("unverified_condition", "Internal electrical integrity unverified."),
            safety_hazards=assessment["safety_hazards"],
            required_ppe=assessment["required_ppe"],
            fault_explanations=assessment["fault_explanations"],
            root_cause_analysis=assessment["root_cause_analysis"],
            recommended_sops=assessment["recommended_sops"],
            replacement_parts=assessment["replacement_parts"],
            llm_reasoning_summary=llm_summary,
            image_path=image_source,
            cv_metadata=cv_obj.model_dump()
        )

    def diagnose_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        context: Optional[ChargerContext] = None,
        station_id: Optional[str] = None,
        threshold: float = 0.5,
        weights_path: Optional[Union[str, Path]] = None
    ) -> DiagnosticResult:
        """
        End-to-end diagnosis directly from an image input.
        Invokes Member 2's predict_fault() then evaluates diagnostics.
        """
        ctx = context or ChargerContext(station_id=station_id)
        if station_id and not ctx.station_id:
            ctx.station_id = station_id

        if predict_fault is None:
            err_result = {
                "status": "error",
                "condition": None,
                "condition_confidence": 0.0,
                "component": None,
                "component_confidence": 0.0,
                "is_fault_detected": None,
                "error_message": "Member 2 inference module (predict_fault) could not be imported."
            }
            return self.diagnose_from_cv(err_result, context=ctx, station_id=ctx.station_id)

        kwargs = {"threshold": threshold}
        if weights_path is not None:
            kwargs["weights_path"] = weights_path

        cv_result = predict_fault(image_input, **kwargs)
        return self.diagnose_from_cv(cv_result, context=ctx, station_id=ctx.station_id)

    def ask(self, diagnostic_result: DiagnosticResult, question: str) -> str:
        """Interactive field technician chat helper."""
        return self.llm_assistant.answer_technician_query(diagnostic_result, question)
