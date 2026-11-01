"""
Pydantic Schemas and Unified Data Contracts for EV Charging Station Diagnostic Engine.
Supports:
  - Computer Vision Dual-Head MobileNetV3 (Condition: healthy/broken, Component: cord/plug/screen/charger)
  - Diagnostic Engine, Explainable Severity Scoring & Evidence Breakdown
  - Research Prototype Work Orders (PDF, HTML, Markdown)
  - Technician Grounded Assistant
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class FaultClass(str, Enum):
    """Standard defect and component classes recognized across detection and classification."""
    DAMAGED_CONNECTOR = "damaged_connector"
    DAMAGED_CABLE = "damaged_cable"
    DAMAGED_SCREEN = "damaged_screen"
    CORROSION_WEAR = "corrosion_wear"
    COMPONENT_PLUG = "plug"
    COMPONENT_CABLE = "cable"
    COMPONENT_SCREEN = "screen"
    COMPONENT_BODY = "body"
    COMPONENT_CHARGER = "charger"
    NORMAL = "normal"
    UNKNOWN = "unknown"


class VisualInspectionStatus(str, Enum):
    """Visual inspection condition strictly determined from optical camera findings."""
    CONFIRMED_FAULT = "POTENTIAL FAULT DETECTED"
    POTENTIAL_FAULT = "POTENTIAL FAULT — VERIFICATION REQUIRED"
    NO_DEFECT_DETECTED = "NO VISIBLE DEFECT DETECTED"
    DISAGREEMENT = "VISIBLE DAMAGE OBSERVED — MODEL CLASSIFICATION DISAGREES"
    INSPECTION_FAILED = "INSPECTION FAILED — MODEL/INPUT ERROR"


class OperationalStatus(str, Enum):
    """Operational telemetry status strictly determined from verified telemetry."""
    UNAVAILABLE = "Operational status unavailable — no telemetry received"
    SIMULATED = "Simulated Telemetry — Not Connected to Real Equipment"
    OPERATIVE = "Operative (Telemetry Verified)"
    FAULTED = "Faulted (Telemetry Verified)"
    RESTRICTED = "Restricted (Telemetry Verified)"


class SeverityLevel(str, Enum):
    """Safety severity classification aligned with electrical hazard potential."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"
    NONE_DETECTED = "NO VISIBLE DAMAGE IDENTIFIED"


class MaintenancePriority(str, Enum):
    """Dispatch urgency category."""
    IMMEDIATE = "IMMEDIATE (Emergency Dispatch < 2-4h)"
    URGENT = "URGENT (Dispatch < 12-24h)"
    SCHEDULED = "SCHEDULED (Dispatch < 36-48h)"
    ROUTINE = "ROUTINE MAINTENANCE"
    PENDING_VERIFICATION = "PENDING ON-SITE VERIFICATION"
    NONE_REQUIRED = "ROUTINE MAINTENANCE"


class ChargerComponent(str, Enum):
    """Standardized component taxonomy."""
    CHARGER = "charger"
    CORD = "cord"
    PLUG = "plug"
    SCREEN = "screen"
    BODY = "body"
    UNKNOWN = "unknown"


class ChargerCondition(str, Enum):
    """Standardized condition taxonomy."""
    HEALTHY = "healthy"
    BROKEN = "broken"


class DetectionItem(BaseModel):
    """Output from object detector with confidence and localization bounding box."""
    class_name: str = Field(..., description="Detected class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0")
    bbox: Optional[List[float]] = Field(default=None, description="[x1, y1, x2, y2] bounding box")
    is_uncertain: bool = Field(default=False, description="True if confidence is below confirmed threshold")
    mapped_category: Optional[str] = Field(default=None, description="Standardized category")


class ChargerContext(BaseModel):
    """Field operating context and environment metadata."""
    station_id: Optional[str] = Field(default=None, description="Charging station identifier")
    charger_type: Optional[str] = Field(default=None, description="Architecture standard (e.g. 150kW DC Fast Charger)")
    power_rating_kw: Optional[float] = Field(default=None, description="Rated power in kW")
    environment: Optional[str] = Field(default=None, description="Depot / roadside / indoor")
    weather_condition: Optional[str] = Field(default=None, description="Ambient weather (e.g. Rain, Clear)")
    has_ocpp_telemetry: bool = Field(default=False, description="Whether live OCPP link is connected")
    is_telemetry_simulated: bool = Field(default=False, description="Whether telemetry is simulated")
    ocpp_state_raw: Optional[str] = Field(default=None, description="Raw OCPP state string if connected")
    inspector_visual_finding: Optional[str] = Field(default=None, description="Inspector-reported observation or defect notice")


class MaintenanceAction(BaseModel):
    """Single standard operating procedure step with safety criticality flag."""
    step_number: int = Field(..., description="Sequential step index")
    title: str = Field(..., description="Action summary title")
    description: str = Field(..., description="Detailed technician instruction")
    is_safety_critical: bool = Field(default=False, description="True if step prevents shock/arc flash")
    action_type: str = Field(default="Inspection", description="Safety / Electrical / Mechanical / Software")


class CVInferenceResult(BaseModel):
    """Structured data contract ingesting predict_fault() output."""
    status: str = Field(..., description="'success' or 'error'")
    condition: Optional[str] = Field(default=None, description="'healthy' or 'broken'")
    condition_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence 0.0 - 1.0")
    component: Optional[str] = Field(default=None, description="'charger', 'cord', 'plug', 'screen'")
    component_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence 0.0 - 1.0")
    is_fault_detected: Optional[bool] = Field(default=None, description="True if broken")
    visual_fault_localized: bool = Field(default=False, description="Classifier boundary: false")
    bounding_box: Optional[Any] = Field(default=None, description="Classifier boundary: None")
    execution_time_ms: Optional[float] = Field(default=None, description="Inference latency in ms")
    model_version: Optional[str] = Field(default=None, description="Model identifier")
    diagnostic_hint: Optional[str] = Field(default=None, description="Diagnostic summary string")
    image_source: Optional[str] = Field(default=None, description="File path or descriptor")
    probabilities: Optional[Dict[str, float]] = Field(default=None, description="Class probabilities")
    error_message: Optional[str] = Field(default=None, description="Error explanation if status is error")


class DiagnosticInput(BaseModel):
    """Complete input payload received by the Diagnostic Engine."""
    station_id: Optional[str] = None
    detections: List[DetectionItem] = Field(default_factory=list, description="List of detected defect items")
    cv_result: Optional[CVInferenceResult] = Field(default=None, description="Parsed CV output")
    context: Optional[ChargerContext] = Field(default_factory=ChargerContext)
    technician_notes: Optional[str] = Field(default=None, description="Optional notes from field inspection")
    inference_error: Optional[str] = Field(default=None, description="Records model/pipeline failures")
    image_path: Optional[str] = None


class DiagnosticResult(BaseModel):
    """Normalized diagnostic result consumed by Dashboard and Reports."""
    report_id: str = Field(default="WO-EVCS-PRELIM")
    station_id: str = Field(default="Not provided")
    charger_type: str = Field(default="Not provided")
    weather_condition: str = Field(default="Not provided")
    visual_inspection_status: VisualInspectionStatus
    operational_status: OperationalStatus
    overall_severity: SeverityLevel
    maintenance_priority: MaintenancePriority
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    component: Optional[str] = None
    condition: Optional[str] = None
    dispatch_sla_hours: Optional[int] = None
    weather_escalated: bool = Field(default=False, description="True if weather elevated severity")
    status_display_label: str = Field(default="Potential Fault Detected")
    is_telemetry_simulated: bool = Field(default=False, description="True if telemetry is simulated")
    inspector_visual_finding: Optional[str] = None
    classification_disagreement: bool = Field(default=False, description="True if visual observation conflicts with model")

    # Three-category observable evidence breakdown
    visible_evidence: str = Field(default="No visible defects observed.")
    possible_explanation: str = Field(default="Appears visually intact from available imagery.")
    unverified_condition: str = Field(default="Internal electrical parameters and functionality remain unverified.")

    safety_hazards: List[str] = Field(default_factory=list)
    required_ppe: List[str] = Field(default_factory=list)
    fault_explanations: List[str] = Field(default_factory=list)
    root_cause_analysis: str = Field(default="Pending qualified physical inspection.")
    recommended_sops: List[MaintenanceAction] = Field(default_factory=list)
    replacement_parts: List[str] = Field(default_factory=list)
    llm_reasoning_summary: Optional[str] = None
    image_path: Optional[str] = None
    disclaimer: str = Field(
        default=(
            "AI-assisted visual inspection only. This result does not certify electrical safety "
            "or confirm internal equipment condition. Damaged charging equipment must be assessed "
            "by qualified personnel using approved procedures before being returned to service."
        )
    )
    standards_referenced: List[str] = Field(
        default_factory=lambda: [
            "IEC 61851-23 (Electric vehicle conductive charging system - Part 23: DC electric vehicle supply equipment)",
            "IEC 62196-3 (Plugs, socket-outlets, vehicle connectors and vehicle inlets - Part 3: Dimensional compatibility)",
            "NFPA 70E (Standard for Electrical Safety in the Workplace)"
        ]
    )
    cv_metadata: Optional[Dict[str, Any]] = None

    @property
    def dispatch_sla_display(self) -> str:
        """Formatted SLA string."""
        if self.dispatch_sla_hours is not None:
            return f"{self.dispatch_sla_hours} Hours"
        return "Not assigned"

    def to_dict(self) -> Dict[str, Any]:
        """Convenience serializer for API and JSON responses."""
        return self.model_dump()
