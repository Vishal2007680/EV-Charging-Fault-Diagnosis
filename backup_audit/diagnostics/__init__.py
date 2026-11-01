"""
Member 3 Diagnostic Assistant Package.
Provides automated fault explanation, severity scoring, SOP recommendations,
and grounded LLM assistant for EV Charging Stations.
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np

from .schema import (
    VisualInspectionStatus,
    OperationalStatus,
    SeverityLevel,
    MaintenancePriority,
    ChargerComponent,
    ChargerCondition,
    ChargerContext,
    MaintenanceAction,
    CVInferenceResult,
    DiagnosticInput,
    DiagnosticResult,
)
from .rules import get_fault_profile, COMPONENT_FAULT_PROFILES
from .status_assessor import StatusAssessor
from .llm_assistant import LLMDiagnosticAssistant
from .report_generator import ReportGenerator
from .engine import EVDiagnosticEngine

# Singleton engine instance for quick API calls
_ENGINE = None


def get_engine(api_key: Optional[str] = None) -> EVDiagnosticEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = EVDiagnosticEngine(api_key=api_key)
    return _ENGINE


def diagnose_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    context: Optional[ChargerContext] = None,
    station_id: str = "EVCS-IND-001",
    threshold: float = 0.5,
    weights_path: Optional[Union[str, Path]] = None,
    api_key: Optional[str] = None
) -> DiagnosticResult:
    """
    Main one-line integration function for Member 4:
    Takes an image, runs Member 2's CV model, and returns full certified diagnosis.
    """
    engine = get_engine(api_key=api_key)
    return engine.diagnose_image(
        image_input=image_input,
        context=context,
        station_id=station_id,
        threshold=threshold,
        weights_path=weights_path
    )


def diagnose_cv_result(
    cv_result: Union[CVInferenceResult, Dict[str, Any]],
    context: Optional[ChargerContext] = None,
    station_id: str = "EVCS-IND-001",
    api_key: Optional[str] = None
) -> DiagnosticResult:
    """
    Main one-line integration function from Member 2's predict_fault() output.
    """
    engine = get_engine(api_key=api_key)
    return engine.diagnose_from_cv(
        cv_data=cv_result,
        context=context,
        station_id=station_id
    )


__all__ = [
    "EVDiagnosticEngine",
    "get_engine",
    "diagnose_image",
    "diagnose_cv_result",
    "ReportGenerator",
    "StatusAssessor",
    "LLMDiagnosticAssistant",
    "VisualInspectionStatus",
    "OperationalStatus",
    "SeverityLevel",
    "MaintenancePriority",
    "ChargerComponent",
    "ChargerCondition",
    "ChargerContext",
    "MaintenanceAction",
    "CVInferenceResult",
    "DiagnosticInput",
    "DiagnosticResult",
    "get_fault_profile",
    "COMPONENT_FAULT_PROFILES",
]
