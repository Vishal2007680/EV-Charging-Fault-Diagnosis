"""
Safety Rule and Standard Operating Procedure (SOP) Directives for EV Charging Diagnostics.
Adheres strictly to evidence-based safety engineering:
  - Distinguishes visible evidence from engineering hypotheses and unverified conditions.
  - Removes universal fabricated electrical limits, torque values, and fixed resistance numbers.
  - Formulates qualified-personnel workflows aligned with approved manufacturer procedures.
  - References applicable standards (IEC 61851-23, IEC 62196-3, NFPA 70E) as advisory technical frameworks.
"""

from typing import Dict, Any, List, Optional
from .schema import (
    SeverityLevel,
    MaintenancePriority,
    MaintenanceAction,
    ChargerComponent,
    ChargerCondition,
    FaultClass,
    DetectionItem,
    ChargerContext,
)


COMPONENT_FAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    ChargerComponent.CORD.value: {
        "display_name": "Charging Cable (Cord)",
        "broken": {
            "name": "Potential Cord Damage Detected",
            "visible_evidence": "Visible outer-jacket damage, surface tear, or sheath breach is suspected on the charging cable.",
            "possible_explanation": "Possible mechanical bending fatigue, drive-over strain, or exterior abrasion compromising the protective outer jacket.",
            "unverified_condition": "Internal conductor continuity, insulation resistance, and dielectric integrity are unverified from optical imagery.",
            "base_severity": SeverityLevel.CRITICAL,
            "base_priority": MaintenancePriority.IMMEDIATE,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "POTENTIAL SHOCK / DIELECTRIC HAZARD: Outer jacket damage may compromise insulation protecting high-voltage DC conductors.",
                "ENVIRONMENTAL INGRESS RISK: Outer jacket rupture allows moisture and contaminant penetration into internal cable layers.",
                "NOTE: Conductor continuity, operating voltage, and insulation resistance have not been measured from visual inspection."
            ],
            "required_ppe": [
                "Appropriate electrical PPE determined by a qualified technician per the site risk assessment and applicable requirements.",
                "Protective eyewear and task-appropriate protective gloves."
            ],
            "replacement_parts": [
                "Compatible manufacturer-approved cable assembly (confirm connector type, cable cross-section, and dispenser model before ordering)."
            ],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Remove from Service",
                    description="Prevent further customer use of the visibly damaged charging cable, apply out-of-service notification, and communicate the hazard to station operators.",
                    is_safety_critical=True,
                    action_type="Safety"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Authorized Electrical Isolation (LOTO)",
                    description="Qualified personnel must isolate upstream electrical power following the installation's approved Lockout/Tagout (LOTO) procedure and verify zero energy before physical contact.",
                    is_safety_critical=True,
                    action_type="Safety"
                ),
                MaintenanceAction(
                    step_number=3,
                    title="Detailed Physical & Electrical Inspection",
                    description="Inspect cable jacket, strain relief, terminations, and connectors using approved service documentation. Follow the manufacturer's approved service procedure; equipment-specific test values are not available from visual imagery.",
                    is_safety_critical=True,
                    action_type="Inspection"
                ),
                MaintenanceAction(
                    step_number=4,
                    title="Repair or Replacement",
                    description="Replace damaged cable assembly with compatible manufacturer-approved parts in accordance with equipment installation specifications.",
                    is_safety_critical=False,
                    action_type="Mechanical"
                ),
                MaintenanceAction(
                    step_number=5,
                    title="Verification & Commissioning",
                    description="Perform all post-repair insulation, bonding, and functional commissioning tests specified by the manufacturer before returning equipment to service.",
                    is_safety_critical=True,
                    action_type="Electrical"
                )
            ]
        },
        "healthy": {
            "name": "No Visible Defect Detected",
            "visible_evidence": "No supported cable outer jacket breach, cuts, or conductor exposure are visible in this image.",
            "possible_explanation": "Cable exterior appears visually continuous and structurally intact in the captured view.",
            "unverified_condition": "Internal conductor fatigue, insulation resistance, and thermal behavior under load remain unverified.",
            "base_severity": SeverityLevel.NONE_DETECTED,
            "base_priority": MaintenancePriority.ROUTINE,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "No visual cable damage detected in this image. Electrical safety and internal functionality have not been verified by visual inspection."
            ],
            "required_ppe": [
                "Standard task-appropriate safety attire and personal protective equipment as determined by facility policy."
            ],
            "replacement_parts": [],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Routine Cable & Holster Inspection",
                    description="Perform scheduled visual inspection of cable resting position, strain relief, and holster retention per manufacturer maintenance schedule.",
                    is_safety_critical=False,
                    action_type="Inspection"
                )
            ]
        }
    },

    ChargerComponent.PLUG.value: {
        "display_name": "Vehicle Connector Plug (Coupler)",
        "broken": {
            "name": "Potential Plug Damage Detected",
            "visible_evidence": "Visible connector damage, housing cracks, or mechanical latch wear are suspected on the coupler head.",
            "possible_explanation": "Possible physical drop impact, mating strain, or localized mechanical stress on connector housing or latch mechanism.",
            "unverified_condition": "Contact condition, interlocks, internal temperature sensors, and electrical performance require physical inspection.",
            "base_severity": SeverityLevel.HIGH,
            "base_priority": MaintenancePriority.URGENT,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "POTENTIAL CONTACT / LATCH HAZARD: Damaged coupler housing or latch mechanism may fail to maintain proper mechanical and electrical coupling.",
                "MOISTURE PENETRATION RISK: Cracks in connector casing may compromise pin weather seals, especially in wet ambient conditions.",
                "NOTE: Contact resistance, pin temperatures, and pilot handshake circuits cannot be measured from optical images."
            ],
            "required_ppe": [
                "Appropriate personal protective equipment selected based on site safety evaluation and the specific task."
            ],
            "replacement_parts": [
                "Compatible manufacturer-approved connector assembly (confirm pin standard, e.g., CCS2, CHAdeMO, or Type 2; parts are not interchangeable)."
            ],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Deactivate Connector & Restrict Use",
                    description="Mark connector unavailable in central management software if integration supports it and action is authorized. Note: software deactivation does not establish electrical isolation.",
                    is_safety_critical=False,
                    action_type="Safety"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Physical & Mechanical Inspection",
                    description="Qualified personnel inspect connector shell, nose cone, latching pawl, contacts, and seals for cracks, distortion, or contamination using approved procedures.",
                    is_safety_critical=True,
                    action_type="Inspection"
                ),
                MaintenanceAction(
                    step_number=3,
                    title="Signal & Continuity Verification",
                    description="Conduct pilot signal, thermistor, and contact checks conditional on connector design and approved manufacturer procedures.",
                    is_safety_critical=True,
                    action_type="Electrical"
                ),
                MaintenanceAction(
                    step_number=4,
                    title="Component Replacement & Commissioning",
                    description="If housing or contacts are damaged, install compatible replacement parts and verify proper operation before returning to service.",
                    is_safety_critical=False,
                    action_type="Mechanical"
                )
            ]
        },
        "healthy": {
            "name": "No Visible Defect Detected",
            "visible_evidence": "Connector housing, latch, and visible contact surfaces show no apparent structural defects in this view.",
            "possible_explanation": "Coupler exterior appears properly seated and undamaged in the captured frame.",
            "unverified_condition": "Internal contact pin spring tension, micro-fissures, and temperature sensor circuits require instrument testing.",
            "base_severity": SeverityLevel.NONE_DETECTED,
            "base_priority": MaintenancePriority.ROUTINE,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "No visual connector defect detected. Electrical safety, contact resistance, and latch interlocks have not been verified by visual inspection."
            ],
            "required_ppe": [
                "Standard facility-approved technician PPE."
            ],
            "replacement_parts": [],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Routine Connector Cleanliness Check",
                    description="Inspect holster and coupler mating face for dirt or debris accumulation in accordance with routine maintenance intervals.",
                    is_safety_critical=False,
                    action_type="Inspection"
                )
            ]
        }
    },

    ChargerComponent.SCREEN.value: {
        "display_name": "HMI Display Screen",
        "broken": {
            "name": "Potential Screen Damage Detected",
            "visible_evidence": "Visible cracks, spiderweb fractures, or display cover surface damage are suspected on the HMI panel.",
            "possible_explanation": "Possible blunt mechanical impact, vandalism, or excessive localized pressure on the display cover.",
            "unverified_condition": "Touch functionality, internal display electronics, and weather sealing have not been tested.",
            "base_severity": SeverityLevel.MEDIUM,
            "base_priority": MaintenancePriority.SCHEDULED,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "POTENTIAL INGRESS RISK: Fractures in the display cover may compromise enclosure weather sealing over time.",
                "USER CUT HAZARD: Damaged cover glass may present a contact injury hazard to equipment users."
            ],
            "required_ppe": [
                "Cut-resistant work gloves and protective safety glasses when handling broken or fractured display glass."
            ],
            "replacement_parts": [
                "Compatible manufacturer-approved display unit and replacement perimeter gasket."
            ],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Access Restriction & Hazard Mitigation",
                    description="If broken glass presents a safety risk, restrict user access and follow the manufacturer's approved isolation and temporary protection procedure.",
                    is_safety_critical=False,
                    action_type="Safety"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Alternative Authorization Review",
                    description="Recommend mobile app, RFID, or automated charging fallback only if the specific charger supports the feature, it is verified functional, and the operator authorizes it.",
                    is_safety_critical=False,
                    action_type="Software"
                ),
                MaintenanceAction(
                    step_number=3,
                    title="Display Replacement & Seal Verification",
                    description="Qualified personnel isolate auxiliary power and replace the display module with a verified compatible assembly, verifying enclosure seal integrity before re-energizing.",
                    is_safety_critical=False,
                    action_type="Mechanical"
                )
            ]
        },
        "healthy": {
            "name": "No Visible Defect Detected",
            "visible_evidence": "No supported glass fracture, screen puncture, or exterior display defects are visible in this image.",
            "possible_explanation": "HMI display cover glass appears clean and intact in this view.",
            "unverified_condition": "Capacitive touch functionality, controller communications, and internal weather seal condition remain unverified.",
            "base_severity": SeverityLevel.NONE_DETECTED,
            "base_priority": MaintenancePriority.ROUTINE,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "No visual screen damage detected. Internal display electronics and touch functionality have not been verified by visual inspection."
            ],
            "required_ppe": [
                "Standard facility-approved PPE."
            ],
            "replacement_parts": [],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Routine Display Maintenance",
                    description="Clean display surface according to manufacturer-approved cleaning guidelines; routine maintenance remains subject to the manufacturer's schedule.",
                    is_safety_critical=False,
                    action_type="Maintenance"
                )
            ]
        }
    },

    ChargerComponent.CHARGER.value: {
        "display_name": "Main Charger Kiosk / Enclosure",
        "broken": {
            "name": "Potential Charger Enclosure Damage Detected",
            "visible_evidence": "Visible cabinet denting, distorted panel alignment, or housing impact damage is suspected on the dispenser enclosure.",
            "possible_explanation": "Possible vehicle collision, external mechanical impact, or enclosure tampering compromising outer cabinet integrity.",
            "unverified_condition": "Internal electrical condition, door interlocks, grounding continuity, and operational status remain unverified.",
            "base_severity": SeverityLevel.HIGH,
            "base_priority": MaintenancePriority.URGENT,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "ENCLOSURE INTEGRITY RISK: Severe cabinet damage or door misalignment may compromise weather sealing or allow unauthorized access to internal areas.",
                "STRUCTURAL RISK: Foundation or cabinet distortion may affect physical stability under cable tension or environmental loads."
            ],
            "required_ppe": [
                "Appropriate personal protective equipment per facility electrical safety procedures and task hazard evaluation."
            ],
            "replacement_parts": [
                "Compatible manufacturer-specified enclosure panels, latches, or seal kits."
            ],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Access Restriction & Site Safety",
                    description="Restrict access around the dispenser if the observed condition presents a potential safety or stability hazard.",
                    is_safety_critical=True,
                    action_type="Safety"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Qualified Enclosure Inspection",
                    description="Have qualified personnel inspect relevant enclosures, doors, latching mechanisms, seals, ventilation, and structural mounting using approved documentation.",
                    is_safety_critical=True,
                    action_type="Inspection"
                ),
                MaintenanceAction(
                    step_number=3,
                    title="Enclosure Repair & Re-Commissioning",
                    description="Repair or replace damaged panels using compatible manufacturer parts and complete approved inspection and commissioning before returning equipment to service.",
                    is_safety_critical=False,
                    action_type="Mechanical"
                )
            ]
        },
        "healthy": {
            "name": "No Visible Defect Detected",
            "visible_evidence": "No supported enclosure breach, severe denting, or pedestal structural defect is visible in this image.",
            "possible_explanation": "Dispenser exterior kiosk appears upright and intact from this angle.",
            "unverified_condition": "Internal electrical condition, busbar connections, ventilation filters, and operational status remain unverified.",
            "base_severity": SeverityLevel.NONE_DETECTED,
            "base_priority": MaintenancePriority.ROUTINE,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "No visual enclosure defect detected in this image. Internal electrical safety, grounding continuity, and ventilation have not been verified by visual inspection."
            ],
            "required_ppe": [
                "Standard facility-approved PPE."
            ],
            "replacement_parts": [],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Routine Enclosure Maintenance",
                    description="Follow scheduled preventive maintenance intervals for cabinet cleanliness, door seals, and ventilation inspection per manufacturer documentation.",
                    is_safety_critical=False,
                    action_type="Maintenance"
                )
            ]
        }
    }
}


# Mapping for YOLO defect classes (backward compatibility)
FAULT_RULE_DETAILS: Dict[str, Dict[str, Any]] = {
    FaultClass.DAMAGED_CABLE.value: {
        "name": "Potential Cord Damage Detected",
        "evidence": "Visible outer jacket breach or surface damage suspected on charging cable.",
        "hypothesis": "Possible mechanical strain, bending fatigue, or abrasion.",
        "unverified": "Internal conductor and dielectric insulation condition are unverified.",
        "ppe": ["Task-appropriate electrical PPE selected by qualified personnel per site risk assessment"],
        "parts": ["Compatible manufacturer-approved cable assembly"],
        "sops": COMPONENT_FAULT_PROFILES[ChargerComponent.CORD.value]["broken"]["sops"]
    },
    FaultClass.DAMAGED_CONNECTOR.value: {
        "name": "Potential Plug Damage Detected",
        "evidence": "Visible connector damage, latch wear, or housing cracks suspected on coupler.",
        "hypothesis": "Possible drop impact or mating strain.",
        "unverified": "Contact resistance, pilot interlocks, and electrical performance require physical testing.",
        "ppe": ["Task-appropriate protective eyewear and protective gloves"],
        "parts": ["Compatible manufacturer-approved connector head assembly (confirm standard: CCS2, CHAdeMO, or Type 2)"],
        "sops": COMPONENT_FAULT_PROFILES[ChargerComponent.PLUG.value]["broken"]["sops"]
    },
    FaultClass.DAMAGED_SCREEN.value: {
        "name": "Potential Screen Damage Detected",
        "evidence": "Visible cracks or surface fractures suspected on display panel.",
        "hypothesis": "Possible mechanical impact or vandalism.",
        "unverified": "Touch responsiveness and internal display electronics unverified.",
        "ppe": ["Cut-resistant gloves and safety glasses"],
        "parts": ["Compatible manufacturer-approved display unit and perimeter gasket"],
        "sops": COMPONENT_FAULT_PROFILES[ChargerComponent.SCREEN.value]["broken"]["sops"]
    },
    FaultClass.CORROSION_WEAR.value: {
        "name": "Surface Wear / Corrosion Observed",
        "evidence": "Visible surface oxidation or non-structural physical wear.",
        "hypothesis": "Possible environmental exposure and weathering.",
        "unverified": "Sub-surface material degradation and electrical continuity unverified.",
        "ppe": ["Standard maintenance PPE"],
        "parts": ["Manufacturer-approved touch-up or seal maintenance materials"],
        "sops": [
            MaintenanceAction(
                step_number=1,
                title="Visual & Mechanical Wear Assessment",
                description="Inspect affected surface for depth of oxidation and verify that structural integrity is unaffected.",
                is_safety_critical=False,
                action_type="Inspection"
            )
        ]
    }
}


def get_fault_profile(component: str, condition: str) -> Dict[str, Any]:
    """Retrieves normalized diagnostic profile for a given component and condition."""
    comp_key = (component or "").lower().strip()
    cond_key = (condition or "").lower().strip()

    if comp_key in COMPONENT_FAULT_PROFILES:
        profile_map = COMPONENT_FAULT_PROFILES[comp_key]
        if cond_key in profile_map:
            return profile_map[cond_key]
        return profile_map.get("broken" if cond_key == "broken" else "healthy", {})

    # Fallback generic profile
    if cond_key == "broken":
        return {
            "name": f"Potential {comp_key.capitalize()} Damage Detected",
            "visible_evidence": f"Visible surface anomaly suspected on component '{comp_key}'.",
            "possible_explanation": "Possible physical damage or mechanical wear.",
            "unverified_condition": "Internal electrical and functional parameters remain unverified.",
            "base_severity": SeverityLevel.HIGH,
            "base_priority": MaintenancePriority.URGENT,
            "dispatch_sla_hours": None,
            "safety_hazards": ["Potential equipment damage requiring professional inspection."],
            "required_ppe": ["Appropriate task-specific PPE per site safety evaluation."],
            "replacement_parts": ["Compatible manufacturer-approved replacement components if damage is confirmed."],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Remove from Service & Inspect",
                    description="Restrict equipment access and arrange physical inspection by qualified personnel.",
                    is_safety_critical=True,
                    action_type="Inspection"
                )
            ]
        }

    return {
        "name": "No Visible Defect Detected",
        "visible_evidence": f"No supported surface defect observed on component '{comp_key}'.",
        "possible_explanation": "Component appears structurally intact from available visual imagery.",
        "unverified_condition": "Internal electrical integrity and operational status remain unverified.",
        "base_severity": SeverityLevel.NONE_DETECTED,
        "base_priority": MaintenancePriority.ROUTINE,
        "dispatch_sla_hours": None,
        "safety_hazards": ["No visible defect detected. Electrical safety has not been verified by visual inspection."],
        "required_ppe": ["Standard facility-approved PPE."],
        "replacement_parts": [],
        "sops": [
            MaintenanceAction(
                step_number=1,
                title="Routine Scheduled Inspection",
                description="Continue regular preventive maintenance schedule per manufacturer documentation.",
                is_safety_critical=False,
                action_type="Inspection"
            )
        ]
    }


class RuleBasedDiagnosticEngine:
    """Evaluates verified YOLO detections and builds structured maintenance directives."""

    @staticmethod
    def evaluate(
        detections: List[DetectionItem],
        context: Optional[ChargerContext] = None,
        inference_error: Optional[str] = None
    ) -> Dict[str, Any]:
        from .status_assessor import StatusAssessor

        ctx = context or ChargerContext()
        assessment = StatusAssessor.assess_status(
            detections=detections,
            context=ctx,
            inference_error=inference_error
        )

        all_sops: List[MaintenanceAction] = []
        all_ppe: List[str] = []
        all_parts: List[str] = []
        fault_explanations: List[str] = []

        faults_to_process = assessment.get("confirmed_faults", []) + assessment.get("uncertain_faults", [])

        for idx, f in enumerate(faults_to_process):
            cls_name = f.class_name.lower().strip()
            details = FAULT_RULE_DETAILS.get(cls_name)
            uncertain_tag = " [POTENTIAL ANOMALY - VERIFY ON SITE]" if f.is_uncertain else ""

            if details:
                fault_explanations.append(
                    f"[{f.class_name.upper()}]{uncertain_tag} (Confidence: {f.confidence:.1%}): "
                    f"Observable: {details.get('evidence', '')} Hypothesis: {details.get('hypothesis', '')}"
                )
                for ppe in details.get("ppe", []):
                    if ppe not in all_ppe:
                        all_ppe.append(ppe)
                for part in details.get("parts", []):
                    if part not in all_parts:
                        all_parts.append(part)
                for sop in details.get("sops", []):
                    sop_copy = sop.model_copy()
                    sop_copy.step_number = len(all_sops) + 1
                    all_sops.append(sop_copy)
            else:
                fault_explanations.append(
                    f"[{f.class_name.upper()}]{uncertain_tag} (Confidence: {f.confidence:.1%}): "
                    f"Visual feature detected without mapped rule profile. Qualified technician inspection required."
                )

        if not faults_to_process:
            all_ppe.append("Standard field technician PPE per facility policy")

        assessment["recommended_sops"] = all_sops or assessment.get("recommended_sops", [])
        assessment["required_ppe"] = all_ppe or assessment.get("required_ppe", [])
        assessment["replacement_parts"] = all_parts or assessment.get("replacement_parts", [])
        if fault_explanations:
            assessment["fault_explanations"] = fault_explanations

        return assessment
