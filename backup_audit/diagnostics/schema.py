"""
Pydantic Schemas and Unified Data Contracts for EV Charging Station Diagnostic Engine.
Supports:
  - Member 1 (YOLO Bounding Box Defect Detections & Annotation Classes)
  - Member 2 (Computer Vision Dual-Head MobileNetV3: Condition & Component)
  - Member 3 (Diagnostic Engine, Severity Scoring, Safety SOPs & Grounded Assistant)
  - Member 4 (Unified Dashboard & Central System Integration)
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class FaultClass(str, Enum):
    """Standard defect and component classes recognized across YOLO & MobileNet."""
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
    CONFIRMED_FAULT = "CONFIRMED FAULT DETECTED"
    POTENTIAL_FAULT = "POTENTIAL FAULT — VERIFICATION REQUIRED"
    NO_DEFECT_DETECTED = "NO SUPPORTED DEFECT DETECTED"
    INSPECTION_FAILED = "INSPECTION FAILED — MODEL/INPUT ERROR"


class OperationalStatus(str, Enum):
    """Operational telemetry status strictly determined from verified OCPP signals."""
    UNAVAILABLE = "Operational status unavailable (no OCPP telemetry)"
    OPERATIVE = "Operative (Telemetry Verified)"
    FAULTED = "Faulted (Telemetry Verified)"
    RESTRICTED = "Restricted (Telemetry Verified)"


class SeverityLevel(str, Enum):
    """Safety severity classification aligned with high-voltage electrical hazards."""
    CRITICAL = "CRITICAL"          # Life safety / direct shock / dielectric failure hazard (SLA < 2-4h)
    HIGH = "HIGH"                  # Urgent mechanical, coupler, or enclosure breach (SLA < 12-24h)
    MEDIUM = "MEDIUM"              # Functional disruption / IP weather seal compromised (SLA < 36-48h)
    LOW = "LOW"                    # Minor cosmetic wear or non-critical surface oxidation (SLA < 72h)
    UNCERTAIN = "UNCERTAIN"        # Low confidence (< 0.65) or inspection error; verification needed
    NONE_DETECTED = "NONE_DETECTED"# Structurally healthy; no visible surface damage detected


class MaintenancePriority(str, Enum):
    """Dispatch urgency category."""
    IMMEDIATE = "IMMEDIATE (Emergency Dispatch < 2-4h)"
    URGENT = "URGENT (Dispatch < 12-24h)"
    SCHEDULED = "SCHEDULED (Dispatch < 36-48h)"
    ROUTINE = "ROUTINE (Next Preventive Service)"
    PENDING_VERIFICATION = "PENDING ON-SITE VERIFICATION"
    NONE_REQUIRED = "NO MAINTENANCE REQUIRED AT THIS TIME"


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
    station_id: str = Field(default="EVCS-001", description="Charging station identifier")
    charger_type: Optional[str] = Field(default="150kW DC Fast Charger (CCS2)", description="Architecture standard")
    power_rating_kw: Optional[float] = Field(default=150.0, description="Rated power in kW")
    environment: Optional[str] = Field(default="Outdoor public highway hub", description="Depot / roadside / indoor")
    weather_condition: Optional[str] = Field(default="Clear / Dry", description="Ambient weather (e.g. Rain, Clear)")
    has_ocpp_telemetry: bool = Field(default=False, description="Whether live OCPP link is connected")
    ocpp_state_raw: Optional[str] = Field(default=None, description="Raw OCPP state string if connected")


class MaintenanceAction(BaseModel):
    """Single standard operating procedure step with safety criticality flag."""
    step_number: int = Field(..., description="Sequential step index")
    title: str = Field(..., description="Action summary title")
    description: str = Field(..., description="Detailed technician instruction")
    is_safety_critical: bool = Field(default=False, description="True if step prevents shock/arc flash")
    action_type: str = Field(default="Inspection", description="Safety / Electrical / Mechanical / Software")


class CVInferenceResult(BaseModel):
    """Structured data contract ingesting Member 2's predict_fault() output."""
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
    station_id: str = "EVCS-001"
    detections: List[DetectionItem] = Field(default_factory=list, description="List of detected defect items")
    cv_result: Optional[CVInferenceResult] = Field(default=None, description="Parsed CV output from Member 2")
    context: Optional[ChargerContext] = Field(default_factory=ChargerContext)
    technician_notes: Optional[str] = Field(default=None, description="Optional notes from field inspection")
    inference_error: Optional[str] = Field(default=None, description="Records model/pipeline failures")
    image_path: Optional[str] = Field(default=None, description="Path to original image")


class DiagnosticResult(BaseModel):
    """Certified final diagnostic output consumed by Member 4's Dashboard."""
    station_id: str
    visual_inspection_status: VisualInspectionStatus
    operational_status: OperationalStatus
    overall_severity: SeverityLevel
    maintenance_priority: MaintenancePriority
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    component: Optional[str] = None
    condition: Optional[str] = None
    dispatch_sla_hours: Optional[int] = None
    weather_escalated: bool = Field(default=False, description="True if weather elevated severity")
    safety_hazards: List[str] = Field(default_factory=list)
    required_ppe: List[str] = Field(default_factory=list)
    fault_explanations: List[str] = Field(default_factory=list)
    root_cause_analysis: str = Field(default="Pending qualified physical inspection.")
    recommended_sops: List[MaintenanceAction] = Field(default_factory=list)
    replacement_parts: List[str] = Field(default_factory=list)
    llm_reasoning_summary: Optional[str] = None
    disclaimer: str = Field(
        default=(
            "RESEARCH PROTOTYPE ADVISORY: Visual analysis alone cannot independently confirm "
            "internal electrical contact resistance, conductor temperature, or electrical safety "
            "compliance. Physical inspection by qualified EV maintenance personnel is required."
        )
    )
    standards_referenced: List[str] = Field(
        default_factory=lambda: [
            "IEC 61851-23 (DC EV Charging Station Safety)",
            "IEC 62196-3 (DC Plugs & Coupler Mechanical Integrity)",
            "NFPA 70E (Electrical Safety in the Workplace)"
        ]
    )
    cv_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convenience serializer for API and JSON responses."""
        return self.model_dump()
