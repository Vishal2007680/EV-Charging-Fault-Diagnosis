"""
Safety Rule and Standard Operating Procedure (SOP) Directives for EV Charging Diagnostics.
Complies with international electrical safety standards:
  - IEC 61851-23: Electric vehicle conductive charging system - DC EV supply equipment
  - IEC 62196-3: Dimensional interchangeability requirements for DC vehicle couplers
  - NFPA 70E: Standard for Electrical Safety in the Workplace (LOTO, Arc Flash, PPE)

Supports both:
  1. Component & Condition profiles (MobileNetV3: cord, plug, screen, charger)
  2. Bounding Box Fault classes (YOLO: damaged_cable, damaged_connector, damaged_screen, etc.)
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
        "display_name": "High-Power Charging Cable (Cord)",
        "broken": {
            "name": "Severed / Frayed Charging Cable Outer Sheath",
            "evidence": "Visible outer jacket breach, deep abrasion, tear, or conductor insulation sleeve exposure.",
            "hypothesis": "Possible high-tension bend fatigue, drive-over mechanical crushing, or severe surface abrasion compromising primary/secondary insulation.",
            "base_severity": SeverityLevel.CRITICAL,
            "base_priority": MaintenancePriority.IMMEDIATE,
            "dispatch_sla_hours": 2,
            "safety_hazards": [
                "CRITICAL LIFE-SAFETY HAZARD: High-voltage DC conductors (up to 1000V DC / 500A) may have compromised dielectric isolation, posing severe electrocution and arc-flash risk.",
                "MOISTURE PENETRATION: Breached cable sheath permits water ingress into internal power cores, creating potential catastrophic phase-to-ground faults.",
                "LIQUID-COOLING LEAKAGE: If liquid-cooled cable, jacket puncture risks dielectric coolant loss and severe thermal runaway under high current."
            ],
            "required_ppe": [
                "Class 0 (1000V) ASTM D120 Insulated Rubber Gloves with Leather Protectors",
                "Arc-Flash Face Shield (minimum 8 cal/cm² NFPA 70E rating)",
                "NFPA 70E Category 2 Flame-Resistant (FR) Coveralls",
                "Dielectric EH-rated Safety Footwear (ASTM F2413)"
            ],
            "replacement_parts": [
                "Factory Pre-Molded DC High-Power Cable Harness Assembly (CCS2/CHAdeMO)",
                "Cable Strain-Relief Retractor Clamp & Balancer Kit",
                "Dielectric Coolant Top-Up Cannister (if liquid-cooled)"
            ],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Lockout / Tagout (LOTO) & Power Isolation",
                    description="Immediately de-energize the upstream 415V/480V AC feeder breaker. Apply physical padlock and 'DANGER - DO NOT OPERATE' tag at the main distribution panel per IEC 61851-23 Clause 7.3.",
                    is_safety_critical=True,
                    action_type="Safety"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Zero-Energy & Dielectric Megohmmeter Test",
                    description="Verify complete absence of voltage with a calibrated CAT IV 1000V multimeter. Conduct a 1000V DC insulation resistance test using a calibrated Megohmmeter (insulation resistance must measure > 1.0 MΩ between conductors and ground).",
                    is_safety_critical=True,
                    action_type="Electrical"
                ),
                MaintenanceAction(
                    step_number=3,
                    title="Cable Harness Replacement & Torque Specification",
                    description="Disconnect damaged cable terminations from the DC busbar. Install replacement factory cable assembly and torque busbar lugs to manufacturer specification (typically 18-22 Nm) using a calibrated torque wrench.",
                    is_safety_critical=False,
                    action_type="Mechanical"
                ),
                MaintenanceAction(
                    step_number=4,
                    title="Protective Earth (PE) Continuity & Commissioning Test",
                    description="Verify Protective Earth (PE) loop resistance is strictly < 0.1 Ω. Perform automated isolation monitoring test via dispenser diagnostic menu before releasing for service.",
                    is_safety_critical=True,
                    action_type="Electrical"
                )
            ]
        },
        "healthy": {
            "name": "Structurally Intact Charging Cable",
            "evidence": "Cable outer sheath is continuous with no visible tears, deep abrasions, cuts, or copper exposure.",
            "hypothesis": "Cable mechanical jacket appears intact and structurally sound.",
            "base_severity": SeverityLevel.NONE_DETECTED,
            "base_priority": MaintenancePriority.NONE_REQUIRED,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "No visible exterior surface damage detected. NOTE: Camera inspection cannot measure internal copper conductor fatigue, contact heating, or insulation resistance."
            ],
            "required_ppe": ["Standard field technician PPE (Safety boots, safety glasses)"],
            "replacement_parts": [],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Routine Cable Retractor & Bend Inspection",
                    description="Inspect cable retractor spring tension to ensure cable does not drag on wet pavement during resting holster storage.",
                    is_safety_critical=False,
                    action_type="Mechanical"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Telemetry Insulation Monitor Check",
                    description="Verify ongoing periodic Insulation Monitoring Device (IMD) telemetry readings on CPMS dashboard.",
                    is_safety_critical=False,
                    action_type="Software"
                )
            ]
        }
    },

    ChargerComponent.PLUG.value: {
        "display_name": "Vehicle Connector Plug (Coupler)",
        "broken": {
            "name": "Damaged Connector Head / Broken Latch / Scorch Marks",
            "evidence": "Visible structural crack on connector nose casing, broken mechanical latch hook, or scorched/bent contact pins.",
            "hypothesis": "Possible mechanical impact from dropping on concrete, drive-off strain while locked into vehicle inlet, or localized resistive heating from contaminated pins.",
            "base_severity": SeverityLevel.HIGH,
            "base_priority": MaintenancePriority.URGENT,
            "dispatch_sla_hours": 12,
            "safety_hazards": [
                "POOR CONTACT RESISTANCE & ARCING: Deformed or oxidized contact pins cause localized resistance heating (>90°C), risking contact weld or fire under high DC amperage.",
                "LATCH FAILURE: A broken mechanical latch cannot maintain the interlock required by IEC 62196-3, potentially interrupting high current under load.",
                "WEATHER INGRESS: Cracked nose housing compromises silicone pin seals, permitting water ingress into power pins."
            ],
            "required_ppe": [
                "Class 0 (1000V) ASTM D120 Insulated Rubber Gloves",
                "Safety Glasses with side shields (ANSI Z87.1)",
                "Static-dissipative safety footwear"
            ],
            "replacement_parts": [
                "Replacement DC Connector Head Kit (CCS2 / CHAdeMO / Type 2)",
                "Mechanical Latch Lever & Return Spring Repair Kit",
                "Silicone Pin-Seal Weather Gasket Pack"
            ],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Holster Lockout & Remote Station Tag",
                    description="Flag connector as 'Unavailable / Inactive' in Central Station Management System (CSMS). Fit an 'OUT OF ORDER' lockout cap over the dispenser holster.",
                    is_safety_critical=False,
                    action_type="Safety"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Contact Pin & Internal Thermistor Diagnostic",
                    description="Inspect DC+, DC-, PE, CP, and PP pins for pit marks or push-back. Measure resistance of internal temperature thermistors (PT1000/NTC) to verify temperature circuit integrity.",
                    is_safety_critical=True,
                    action_type="Electrical"
                ),
                MaintenanceAction(
                    step_number=3,
                    title="Mechanical Latch & Proximity Switch Verification",
                    description="Verify latch return spring force. Test Proximity Pilot (PP) switch resistance toggles between 480 Ω (unpressed) and 150 Ω (pressed) per IEC 62196-3.",
                    is_safety_critical=False,
                    action_type="Mechanical"
                ),
                MaintenanceAction(
                    step_number=4,
                    title="EV Simulator Pilot Handshake Validation",
                    description="Attach a calibrated EVSE tester to confirm standard 4-state pilot handshake sequence (State A -> B -> C -> D) prior to releasing for public charging.",
                    is_safety_critical=True,
                    action_type="Electrical"
                )
            ]
        },
        "healthy": {
            "name": "Structurally Intact Connector Plug",
            "evidence": "Plug nozzle, mechanical latch, and visible contact pins appear clean, intact, and properly aligned.",
            "hypothesis": "Connector head shows good physical condition with intact mechanical latching mechanism.",
            "base_severity": SeverityLevel.NONE_DETECTED,
            "base_priority": MaintenancePriority.NONE_REQUIRED,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "No visible exterior plug damage detected. NOTE: Internal pin spring tension, micro-pitting, and temperature sensors require physical instrument testing."
            ],
            "required_ppe": ["Standard field technician PPE (Safety boots, safety glasses)"],
            "replacement_parts": [],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Connector Holster & Pin Hygiene Check",
                    description="Inspect holster drainage port for debris and ensure connector nozzle seats securely when holstered.",
                    is_safety_critical=False,
                    action_type="Inspection"
                )
            ]
        }
    },

    ChargerComponent.SCREEN.value: {
        "display_name": "HMI Touch Display Screen",
        "broken": {
            "name": "Damaged / Shattered HMI Display Panel",
            "evidence": "Visible spiderweb crack, shattered cover glass, impact puncture, or shattered LCD screen matrix.",
            "hypothesis": "Possible vandalism or blunt mechanical impact that breaches enclosure ingress protection (IP54 / IP55).",
            "base_severity": SeverityLevel.MEDIUM,
            "base_priority": MaintenancePriority.SCHEDULED,
            "dispatch_sla_hours": 36,
            "safety_hazards": [
                "ENCLOSURE WEATHER SEAL COMPROMISED: Rain and dust can infiltrate through glass fissures into low-voltage controller and RFID reader electronics.",
                "TECHNICIAN / USER CUT HAZARD: Exposed fractured tempered glass shards pose laceration danger to drivers and technicians."
            ],
            "required_ppe": [
                "ANSI Level A4 Cut-Resistant Gloves",
                "ANSI Z87.1 Safety Glasses with side shields"
            ],
            "replacement_parts": [
                "Replacement Industrial Touch Display Assembly (OEM spec, anti-glare, IP65 front)",
                "Weatherproof EPDM Bezel Perimeter Gasket Seal"
            ],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Weather-Seal Triage Film Application",
                    description="Immediately apply an IP-rated clear self-adhesive protective barrier membrane over shattered glass to prevent rainwater infiltration.",
                    is_safety_critical=False,
                    action_type="Mechanical"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Enable App-Only / Autocharge Fallback",
                    description="Configure dispenser via CPMS to bypass screen touch input, enabling drivers to start charging via mobile application, RFID card, or Autocharge (Plug & Charge).",
                    is_safety_critical=False,
                    action_type="Software"
                ),
                MaintenanceAction(
                    step_number=3,
                    title="Display Module Replacement & Bezel Resealing",
                    description="Isolate 24V auxiliary power. Disconnect HDMI/LVDS and USB touch cables. Install replacement screen assembly and torque bezel fasteners to ensure waterproof gasket compression.",
                    is_safety_critical=False,
                    action_type="Mechanical"
                )
            ]
        },
        "healthy": {
            "name": "Intact HMI Display Screen",
            "evidence": "Display cover glass is smooth, intact, with no visible cracks, chips, or moisture pooling behind glass.",
            "hypothesis": "HMI display module appears physically sound with intact IP ingress weather seal.",
            "base_severity": SeverityLevel.NONE_DETECTED,
            "base_priority": MaintenancePriority.NONE_REQUIRED,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "No visual screen damage detected. NOTE: Touch capacitive sensitivity and backlight health require active operational testing."
            ],
            "required_ppe": ["Standard field technician PPE (Safety boots, safety glasses)"],
            "replacement_parts": [],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Routine Screen Cleaning",
                    description="Wipe display surface with non-abrasive anti-static screen cleaner to maintain visibility.",
                    is_safety_critical=False,
                    action_type="Maintenance"
                )
            ]
        }
    },

    ChargerComponent.CHARGER.value: {
        "display_name": "Main Charger Kiosk / Cabinet Enclosure",
        "broken": {
            "name": "Damaged Cabinet Enclosure / Breached Housing",
            "evidence": "Visible collision dent, punctured enclosure sheet metal, loose/unlocked access door, or fractured anchor pedestal.",
            "hypothesis": "Possible vehicle impact against dispenser, severe environmental damage, or forced entry tampering.",
            "base_severity": SeverityLevel.HIGH,
            "base_priority": MaintenancePriority.URGENT,
            "dispatch_sla_hours": 18,
            "safety_hazards": [
                "ACCESS TO HAZARDOUS LIVE PARTS: Damaged door or cabinet opening allows unauthorized public or wildlife contact with 415V AC / 1000V DC internal power busbars.",
                "STRUCTURAL COLLAPSE RISK: Pedestal or anchor damage risks dispenser tipping during high wind or vehicle cable strain.",
                "MASSIVE WATER INGRESS: Enclosure breach allows direct rain penetration onto power converters, rectifiers, and auxiliary transformers."
            ],
            "required_ppe": [
                "Standard High-Voltage Electrician PPE",
                "Class 0 (1000V) Insulated Gloves",
                "Safety Helmet with face shield",
                "Steel-toe EH safety boots"
            ],
            "replacement_parts": [
                "Replacement Cabinet Door Panel with Keyed Latch Mechanism",
                "High-Strength Concrete Anchor Bolt Set (M16 / M20)",
                "Silicone Weather Seal Gasket Strip"
            ],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Bay Cordoning & Emergency Disconnect",
                    description="Place high-visibility safety cones and caution tape around the entire charging bay. Engage the Emergency Stop (E-Stop) push-button.",
                    is_safety_critical=True,
                    action_type="Safety"
                ),
                MaintenanceAction(
                    step_number=2,
                    title="Cabinet Earthing Bond & Door Tamper Verification",
                    description="Verify that enclosure chassis earth bonding conductor is intact (< 0.1 Ω to earth ground). Test door tamper safety interlock microswitches.",
                    is_safety_critical=True,
                    action_type="Electrical"
                ),
                MaintenanceAction(
                    step_number=3,
                    title="Enclosure Repair & Weatherproofing",
                    description="Re-align or replace warped door panels. Replace damaged perimeter weatherstripping to restore IP54 / NEMA 3R enclosure ingress rating.",
                    is_safety_critical=False,
                    action_type="Mechanical"
                )
            ]
        },
        "healthy": {
            "name": "Structurally Sound Charger Kiosk",
            "evidence": "Cabinet enclosure is clean, doors are locked shut, and base anchor pedestal appears securely fixed.",
            "hypothesis": "Dispenser housing appears mechanically sound with intact weather seal and structural integrity.",
            "base_severity": SeverityLevel.NONE_DETECTED,
            "base_priority": MaintenancePriority.NONE_REQUIRED,
            "dispatch_sla_hours": None,
            "safety_hazards": [
                "No visual enclosure damage detected. NOTE: Internal thermal ventilation, filter clogging, and surge arrestor status require physical inspection."
            ],
            "required_ppe": ["Standard field technician PPE (Safety boots, safety glasses)"],
            "replacement_parts": [],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Routine Ventilation Grille Inspection",
                    description="Inspect intake and exhaust air filtration grilles for dust/debris buildup to preserve converter cooling airflow.",
                    is_safety_critical=False,
                    action_type="Maintenance"
                )
            ]
        }
    }
}


# Mapping for YOLO defect classes
FAULT_RULE_DETAILS: Dict[str, Dict[str, Any]] = {
    FaultClass.DAMAGED_CABLE.value: {
        "name": "Damaged / Severed Cable Sheath",
        "evidence": "Visible tear, split, cut, or abrasion on charging cable outer insulation sleeve.",
        "hypothesis": "Possible mechanical bend fatigue, vehicle run-over, or deep abrasion compromising secondary insulation.",
        "ppe": [
            "Class 0 (1000V) ASTM D120 Insulated Rubber Gloves + Leather Protectors",
            "Arc-Flash Face Shield (minimum 8 cal/cm² rating)",
            "NFPA 70E Category 2 Flame-Resistant (FR) Clothing",
            "Dielectric EH-rated Safety Footwear"
        ],
        "parts": [
            "Replacement High-Power Cable Harness Assembly with factory molded connector",
            "Cable strain-relief retractor clamp kit"
        ],
        "sops": [
            MaintenanceAction(
                step_number=1,
                title="Lockout / Tagout (LOTO) & Power Isolation",
                description="Immediately trip upstream AC circuit breaker and apply physical LOTO padlock at the distribution board before inspecting cable.",
                is_safety_critical=True,
                action_type="Safety"
            ),
            MaintenanceAction(
                step_number=2,
                title="Zero-Energy & Dielectric Verification",
                description="Verify absence of voltage with a calibrated CAT IV 1000V meter. Perform 1000V DC megohmmeter insulation resistance test (must measure > 1.0 MΩ).",
                is_safety_critical=True,
                action_type="Electrical"
            ),
            MaintenanceAction(
                step_number=3,
                title="Cable Assembly Replacement",
                description="Replace damaged cable assembly if conductors are exposed or outer jacket breach exceeds manufacturer repair limits.",
                is_safety_critical=False,
                action_type="Mechanical"
            ),
            MaintenanceAction(
                step_number=4,
                title="Earth Continuity Validation",
                description="Verify Protective Earth (PE) loop resistance is < 0.1 Ω prior to re-energizing the dispenser.",
                is_safety_critical=True,
                action_type="Electrical"
            )
        ]
    },
    FaultClass.DAMAGED_CONNECTOR.value: {
        "name": "Damaged Connector Head / Broken Latch / Deformed Pins",
        "evidence": "Visible structural crack on plug nozzle, damaged mechanical latch lever, or scorched/misaligned pins.",
        "hypothesis": "Possible high mechanical drop impact, mating strain, or poor contact seating causing localized heating.",
        "ppe": [
            "Class 0 (1000V) Insulated Gloves",
            "Safety Glasses with side shields",
            "Static-dissipative safety footwear"
        ],
        "parts": [
            "Replacement CCS2 / CHAdeMO / Type-2 Connector Head Assembly",
            "Mechanical Latch Return Spring & Microswitch Kit",
            "Silicone pin-seal weather gasket pack"
        ],
        "sops": [
            MaintenanceAction(
                step_number=1,
                title="De-activate Dispenser & Tag Holster",
                description="Flag station as out of service in management backend and place 'Out of Service' protective lock-cap on connector holster.",
                is_safety_critical=False,
                action_type="Safety"
            ),
            MaintenanceAction(
                step_number=2,
                title="Contact Pin & Thermal Sensor Inspection",
                description="Inspect power pins (DC+, DC-, PE, CP, PP) for scorch marks, pitting, or push-back. Check internal temperature thermistor continuity.",
                is_safety_critical=True,
                action_type="Electrical"
            ),
            MaintenanceAction(
                step_number=3,
                title="Mechanical Latch & Proximity Switch Verification",
                description="Verify latch return spring force and ensure Proximity Pilot (PP) switch resistance toggles correctly per IEC 62196-3.",
                is_safety_critical=False,
                action_type="Mechanical"
            ),
            MaintenanceAction(
                step_number=4,
                title="EV Simulator Calibration Test",
                description="Connect an EVSE test simulator to confirm proper 4-state pilot handshake (A -> B -> C -> D) before releasing for customer charging.",
                is_safety_critical=True,
                action_type="Electrical"
            )
        ]
    },
    FaultClass.DAMAGED_SCREEN.value: {
        "name": "Damaged / Cracked HMI Display",
        "evidence": "Visible spiderweb crack, shattered glass, or impact puncture on dispenser user interface screen.",
        "hypothesis": "Possible vandalism or blunt mechanical impact. IP weather seal integrity compromised.",
        "ppe": [
            "Cut-resistant gloves (ANSI Level A4+)",
            "Safety Glasses with side protection"
        ],
        "parts": [
            "Replacement Industrial Touch Display Unit (OEM specification)",
            "Weatherproof EPDM Bezel Perimeter Gasket"
        ],
        "sops": [
            MaintenanceAction(
                step_number=1,
                title="Weather Protection Triage",
                description="Apply an IP-rated clear weather-tight adhesive patch over cracked glass to prevent rain infiltration onto internal logic boards.",
                is_safety_critical=False,
                action_type="Mechanical"
            ),
            MaintenanceAction(
                step_number=2,
                title="Enable App-Only Fallback",
                description="Configure dispenser for mobile-app or RFID tap-to-start charging so drivers can charge without touch display interaction.",
                is_safety_critical=False,
                action_type="Software"
            ),
            MaintenanceAction(
                step_number=3,
                title="HMI Display Module Replacement",
                description="Isolate low-voltage control supply, disconnect LVDS/HDMI and USB ribbon cables, and swap out display assembly.",
                is_safety_critical=False,
                action_type="Mechanical"
            )
        ]
    },
    FaultClass.CORROSION_WEAR.value: {
        "name": "Visible Corrosion / Structural Wear",
        "evidence": "Visible oxidation, paint blistering, or rust on dispenser housing or metal bracketry.",
        "hypothesis": "Environmental exposure, road salt, or moisture pooling on metal joints.",
        "ppe": [
            "General maintenance work gloves",
            "Dust mask and safety glasses"
        ],
        "parts": [
            "Anti-corrosion zinc primer and paint touch-up kit",
            "Replacement fastener hardware (marine grade 316 stainless)"
        ],
        "sops": [
            MaintenanceAction(
                step_number=1,
                title="Surface Treatment & Enclosure Inspection",
                description="Remove loose oxidation with wire brush, treat with rust converter, and apply outdoor protective coating.",
                is_safety_critical=False,
                action_type="Mechanical"
            )
        ]
    }
}


def get_fault_profile(component: str, condition: str) -> Dict[str, Any]:
    """Retrieves standardized diagnostic rule profile for a component and condition."""
    comp_key = (component or "").lower().strip()
    cond_key = (condition or "").lower().strip()

    comp_dict = COMPONENT_FAULT_PROFILES.get(comp_key)
    if not comp_dict:
        return {
            "display_name": f"Component ({component})",
            "name": f"Unmapped Component ({component}) - {condition}",
            "evidence": f"Visual detection on component '{component}' with condition '{condition}'.",
            "hypothesis": "Uncataloged component feature observed. Physical inspection recommended.",
            "base_severity": SeverityLevel.UNCERTAIN if cond_key == "broken" else SeverityLevel.NONE_DETECTED,
            "base_priority": MaintenancePriority.PENDING_VERIFICATION if cond_key == "broken" else MaintenancePriority.NONE_REQUIRED,
            "dispatch_sla_hours": 48 if cond_key == "broken" else None,
            "safety_hazards": ["Unclassified component detected; on-site qualified technician verification advised."],
            "required_ppe": ["Standard field technician PPE"],
            "replacement_parts": [],
            "sops": [
                MaintenanceAction(
                    step_number=1,
                    title="Visual & Mechanical Verification",
                    description="Perform on-site inspection of component to evaluate physical integrity.",
                    is_safety_critical=False,
                    action_type="Inspection"
                )
            ]
        }

    return comp_dict.get(cond_key, comp_dict.get("healthy"))


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

        faults_to_process = assessment["confirmed_faults"] + assessment["uncertain_faults"]

        for idx, f in enumerate(faults_to_process):
            cls_name = f.class_name.lower().strip()
            details = FAULT_RULE_DETAILS.get(cls_name)
            uncertain_tag = " [POTENTIAL ANOMALY - VERIFY ON SITE]" if f.is_uncertain else ""

            if details:
                fault_explanations.append(
                    f"[{f.class_name.upper()}]{uncertain_tag} (Confidence: {f.confidence:.1%}): "
                    f"Observable: {details['evidence']} Hypothesis: {details['hypothesis']}"
                )
                for ppe in details["ppe"]:
                    if ppe not in all_ppe:
                        all_ppe.append(ppe)
                for part in details["parts"]:
                    if part not in all_parts:
                        all_parts.append(part)
                for sop in details["sops"]:
                    sop_copy = sop.model_copy()
                    sop_copy.step_number = len(all_sops) + 1
                    all_sops.append(sop_copy)
            else:
                fault_explanations.append(
                    f"[{f.class_name.upper()}]{uncertain_tag} (Confidence: {f.confidence:.1%}): "
                    f"Visual feature detected without mapped rule profile. Qualified technician inspection required."
                )

        if not faults_to_process:
            all_ppe.append("Standard field technician PPE (safety boots, safety glasses)")

        assessment["recommended_sops"] = all_sops
        assessment["required_ppe"] = all_ppe
        assessment["replacement_parts"] = all_parts
        assessment["fault_explanations"] = fault_explanations

        return assessment
