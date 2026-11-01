"""
EV Charging Station Diagnostic Assistant — Member 3 Streamlit Web Application.
Neuro-Symbolic Architecture:
  - MobileNetV3-Small Dual-Head CV Inference Ingestion
  - Evidence-Based Severity Assessment & Dispatch SLA Gating
  - Grounded Multimodal Technician Assistant (Offline & Online)
  - Standardized Maintenance Work Order Generation (PDF, HTML, Markdown)
  - Fully responsive layout with wrapped metrics (no text clipping)
"""

import sys
import os
from pathlib import Path
from PIL import Image
import streamlit as st

# Configure page properties
st.set_page_config(
    page_title="EV Charging Diagnostic Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.diagnostics import (
    EVDiagnosticEngine,
    ChargerContext,
    ReportGenerator,
    SeverityLevel,
    VisualInspectionStatus,
    OperationalStatus,
    MaintenancePriority,
)

# Custom Responsive CSS overrides (strictly eliminates truncation & unwanted ellipses)
st.markdown(
    """
    <style>
    /* Metric Card Wrapping - Prevent truncation across all screen sizes */
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
        font-size: 1.15rem !important;
        line-height: 1.3 !important;
    }
    div[data-testid="stMetricLabel"] > div {
        white-space: normal !important;
        overflow: visible !important;
        font-size: 0.85rem !important;
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* Custom Flex/Grid Metric Container */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 12px;
        margin: 14px 0 20px 0;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.35;
        word-break: break-word;
    }

    /* Observable Evidence Card */
    .evidence-card {
        background: #f1f5f9;
        border-left: 4px solid #0284c7;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0;
        font-size: 0.95em;
        line-height: 1.5;
    }

    /* Mandatory Safety Disclaimer Banner */
    .safety-disclaimer-banner {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        color: #92400e;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 14px 0;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if "engine" not in st.session_state:
    st.session_state.engine = EVDiagnosticEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title & Header
st.title("⚡ Vision-Language AI Assistant for EV Charging Stations")
st.subheader("Intelligent Fault Diagnosis, Severity Scoring & SOP Engine")
st.caption("Neuro-Symbolic Diagnostic Architecture: Evidence-Based Rules + Grounded Multimodal Assistant")
st.markdown("---")

# ------------------------------------------------------------------
# Sidebar: Configuration & Image Source
# ------------------------------------------------------------------
st.sidebar.header("🛠️ Station Operating Context")
raw_station_id = st.sidebar.text_input("Station Identifier", value="EVCS-IND-PUNE-01", help="Leave blank if unassigned")
station_id = raw_station_id.strip() if raw_station_id.strip() else None

raw_charger_type = st.sidebar.selectbox(
    "Hardware Architecture",
    [
        "150kW DC Fast Charger (CCS2 Dual Gun)",
        "350kW Ultra-Fast Liquid-Cooled Dispenser (CCS2)",
        "50kW DC Fast Charger (CCS2 / CHAdeMO)",
        "22kW AC Commercial Dispenser (Type 2)",
        "Not provided / Unverified"
    ]
)
charger_type = None if raw_charger_type == "Not provided / Unverified" else raw_charger_type

raw_weather = st.sidebar.selectbox(
    "Ambient Weather Condition",
    [
        "Clear / Dry",
        "Monsoon Rain / Wet Ground",
        "High Humidity / Fog",
        "Extreme Heat (>45°C)",
        "Not provided / Unverified"
    ]
)
weather = None if raw_weather == "Not provided / Unverified" else raw_weather

# Inspector Visual Observation (Physical visual findings / Disagreement detection)
inspector_obs = st.sidebar.selectbox(
    "Inspector Visual Observation",
    [
        "Automated Camera Analysis Only",
        "Visible Damage Observed (Shattered / Fractured Display Glass)",
        "Visible Damage Observed (Torn Cable Sheath / Exposed Wires)",
        "Visible Damage Observed (Damaged Coupler / Latch)",
        "Visible Damage Observed (Cabinet Enclosure Dents / Impact)",
        "No Visible Physical Damage Observed"
    ],
    help="Record human inspector physical observations to verify or catch classifier discrepancies"
)
inspector_finding = None
if inspector_obs not in ["Automated Camera Analysis Only", "No Visible Physical Damage Observed"]:
    inspector_finding = inspector_obs

# Telemetry Configuration
has_ocpp = st.sidebar.checkbox("Simulate Telemetry Stream (Demo Simulation)", value=False)
is_simulated = has_ocpp
ocpp_raw = "Operative" if has_ocpp else None
if has_ocpp:
    st.sidebar.caption("⚠️ Telemetry is simulated for demonstration and does not represent a live connected station.")

st.sidebar.markdown("---")
st.sidebar.header("📷 Input Station Image")

source_mode = st.sidebar.radio(
    "Select Image Input Mode",
    ["Test Dataset Library", "Upload Custom Image"]
)

def get_dataset_base() -> Path:
    candidates = [
        PROJECT_ROOT / "dataset" / "flare_classification",
        Path("C:/Users/INDIA/Desktop/ProjectWork/ProjectWork/dataset/flare_classification"),
        Path("C:/Users/INDIA/Desktop/EV_Charging_AI_Dataset_Handover/flare_classification"),
        Path("C:/Users/INDIA/Desktop/EV_Charging/dataset/flare_classification"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return PROJECT_ROOT / "dataset" / "flare_classification"

dataset_base = get_dataset_base()
sample_images = {
    "Broken Cable Sheath (cord / broken)": dataset_base / "cord" / "broken" / "cord_broken_0065.jpg",
    "Damaged Coupler Plug (plug / broken)": dataset_base / "plug" / "broken" / "plug_broken_0113.jpg",
    "Shattered Touch Display (screen / broken)": dataset_base / "screen" / "broken" / "screen_broken_0144.jpg",
    "Illuminated Fractured Screen (screen / broken false-negative)": dataset_base / "screen" / "broken" / "screen_broken_0149.jpg",
    "Dented Cabinet Housing (charger / broken)": dataset_base / "charger" / "broken" / "charger_broken_0000.jpg",
    "Healthy Charging Cable (cord / healthy)": dataset_base / "cord" / "healthy" / "cord_healthy_0094.jpg",
    "Healthy Connector Plug (plug / healthy)": dataset_base / "plug" / "healthy" / "plug_healthy_0130.jpg",
    "Healthy Display Screen (screen / healthy)": dataset_base / "screen" / "healthy" / "screen_healthy_0174.jpg",
    "Healthy Charger Kiosk (charger / healthy)": dataset_base / "charger" / "healthy" / "charger_healthy_0023.jpg",
}

active_image = None
active_image_desc = ""
active_image_file_path = None

if source_mode == "Test Dataset Library":
    selected_sample = st.sidebar.selectbox("Choose Verified Sample Image", list(sample_images.keys()))
    img_path = sample_images[selected_sample]
    if img_path.exists():
        active_image = Image.open(img_path).convert("RGB")
        active_image_desc = str(img_path.name)
        active_image_file_path = str(img_path)
    else:
        st.sidebar.error(f"Sample file not found: {img_path}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload station photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file).convert("RGB")
        active_image_desc = uploaded_file.name
        # Persist uploaded image for PDF generator
        temp_img_dir = PROJECT_ROOT / "temp_uploads"
        temp_img_dir.mkdir(parents=True, exist_ok=True)
        temp_img_path = temp_img_dir / uploaded_file.name
        active_image.save(temp_img_path)
        active_image_file_path = str(temp_img_path)

threshold = st.sidebar.slider("Broken Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
st.sidebar.caption("Sensitivity threshold for classification head. Lower values increase defect recall.")

# ------------------------------------------------------------------
# Main Diagnostic Execution
# ------------------------------------------------------------------
if active_image is None:
    st.info("👈 Please select or upload an image from the sidebar to begin diagnosis.")
    st.stop()

# Build Context
ctx = ChargerContext(
    station_id=station_id,
    charger_type=charger_type,
    weather_condition=weather,
    has_ocpp_telemetry=has_ocpp,
    is_telemetry_simulated=is_simulated,
    ocpp_state_raw=ocpp_raw,
    inspector_visual_finding=inspector_finding
)

# Run Inference & Diagnostic Evaluation
with st.spinner("Running MobileNetV3-Small Dual-Head CV Inference & Diagnostic Engine..."):
    diag_result = st.session_state.engine.diagnose_image(
        active_image,
        context=ctx,
        station_id=station_id,
        threshold=threshold
    )
    if active_image_file_path:
        diag_result.image_path = active_image_file_path

# ------------------------------------------------------------------
# Visual Presentation
# ------------------------------------------------------------------
col_img, col_cv = st.columns([1, 1.2])

with col_img:
    st.image(active_image, caption=f"Station Image: {active_image_desc}", use_container_width=True)

with col_cv:
    st.markdown("### 🔍 Computer Vision Telemetry & Classification")
    meta = diag_result.cv_metadata or {}

    cond_color = "#d32f2f" if diag_result.condition == "broken" else "#2e7d32"
    st.markdown(
        f"""
        <div style="background:#f4f6f8; padding:12px 16px; border-radius:6px; margin-bottom:12px;">
            <p style="margin:0 0 6px 0;"><strong>Condition Prediction:</strong> <span style="color:{cond_color}; font-weight:bold; font-size:1.1em;">{str(diag_result.condition).upper()}</span></p>
            <p style="margin:0 0 6px 0;"><strong>Component Inspected:</strong> <span style="font-weight:bold; font-size:1.1em;">{str(diag_result.component).upper()}</span></p>
            <p style="margin:0 0 6px 0;"><strong>Inference Latency:</strong> <code>{meta.get('execution_time_ms', 'N/A')} ms</code></p>
            <p style="margin:0;"><strong>Model Architecture:</strong> <code>{meta.get('model_version', 'MobileNetV3-Small-DualHead-v1')}</code></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"**Model Condition Score ({diag_result.condition}):** {diag_result.confidence_score*100:.1f}%")
    st.progress(float(diag_result.confidence_score))
    st.caption("ℹ️ Confidence reflects the model's pattern prediction score, not a probability that the equipment is electrically safe.")

    comp_conf = float(meta.get("component_confidence", 0.0))
    st.write(f"**Component Classification Score ({diag_result.component}):** {comp_conf*100:.1f}%")
    st.progress(comp_conf)

    if diag_result.confidence_score < 0.65:
        st.warning(f"⚠️ Low AI classification score ({diag_result.confidence_score*100:.1f}% < 65%). On-site technician physical verification required.")

# ------------------------------------------------------------------
# Safety Warning & Status HUD Banner
# ------------------------------------------------------------------
sev_styles = {
    SeverityLevel.CRITICAL: ("#ffebee", "#c62828", "🔴"),
    SeverityLevel.HIGH: ("#fff3e0", "#ef6c00", "🟠"),
    SeverityLevel.MEDIUM: ("#fffde7", "#f57f17", "🟡"),
    SeverityLevel.LOW: ("#e3f2fd", "#1565c0", "🔵"),
    SeverityLevel.UNCERTAIN: ("#f3e5f5", "#7b1fa2", "🟣"),
    SeverityLevel.NONE_DETECTED: ("#e8f5e9", "#2e7d32", "🟢"),
}
bg_color, text_color, icon = sev_styles.get(diag_result.overall_severity, ("#f5f5f5", "#333", "⚪"))

weather_text = " — 🌧️ **ESCALATED TO CRITICAL DUE TO ACTIVE RAIN**" if diag_result.weather_escalated else ""

# Disagreement Callout Banner
if diag_result.classification_disagreement:
    st.markdown(
        f"""
        <div style="background-color: #fef2f2; border-left: 6px solid #ef4444; padding: 14px 20px; border-radius: 4px; margin: 12px 0;">
            <h4 style="color: #b91c1c; margin: 0 0 6px 0;">⚠️ CLASSIFICATION DISAGREEMENT OBSERVED</h4>
            <span style="color: #444;">Physical observation reported <strong>{diag_result.inspector_visual_finding}</strong>, but 2D vision model predicted <strong>HEALTHY ({diag_result.confidence_score*100:.1f}%)</strong>. Both observations are preserved. Equipment must be isolated and physically inspected by a qualified technician before energization.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    f"""
    <div style="background-color: {bg_color}; border-left: 6px solid {text_color}; padding: 14px 20px; border-radius: 4px; margin: 18px 0;">
        <h3 style="color: {text_color}; margin: 0 0 6px 0;">{icon} {diag_result.status_display_label}{weather_text}</h3>
        <span style="color: #444;">Rationale: <strong>{diag_result.root_cause_analysis}</strong></span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="safety-disclaimer-banner">
        <strong>⚠️ MANDATORY SAFETY ADVISORY:</strong> AI-assisted visual inspection only. This result does not certify electrical safety or confirm internal equipment condition. Damaged charging equipment must be assessed by qualified personnel using approved procedures before being returned to service.
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------
# Responsive Metric Badges (No Text Truncation)
# ------------------------------------------------------------------
sla_text = diag_result.dispatch_sla_display
op_label = diag_result.operational_status.value
if diag_result.is_telemetry_simulated:
    op_label += " [SIMULATED]"

st.markdown(
    f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">Visual Inspection Status</div>
            <div class="metric-value">{diag_result.visual_inspection_status.value}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Safety Severity Level</div>
            <div class="metric-value">{diag_result.overall_severity.value}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Maintenance Priority</div>
            <div class="metric-value">{diag_result.maintenance_priority.value}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Dispatch SLA</div>
            <div class="metric-value">{sla_text}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------
# Diagnostic Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Observable Evidence & SOPs",
    "🛡️ Safety Hazards & PPE",
    "🤖 Field Technician Assistant (Q&A)",
    "📄 Work Order Reports (PDF / HTML)"
])

with tab1:
    st.subheader("🔍 Observable Evidence & Diagnostic Breakdown")
    st.markdown(
        f"""
        <div class="evidence-card">
            <p style="margin: 0 0 8px 0;"><strong>Visible Evidence:</strong> {diag_result.visible_evidence}</p>
            <p style="margin: 0 0 8px 0;"><strong>Possible Explanation:</strong> {diag_result.possible_explanation}</p>
            <p style="margin: 0;"><strong>Unverified Condition:</strong> {diag_result.unverified_condition}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📋 Component-Specific Standard Operating Procedures (SOP)")
    if diag_result.recommended_sops:
        for sop in diag_result.recommended_sops:
            badge = "🚨 **[SAFETY CRITICAL]**" if sop.is_safety_critical else f"ℹ️ [{sop.action_type.upper()}]"
            with st.expander(f"Step {sop.step_number}: {sop.title} {badge}", expanded=sop.is_safety_critical):
                st.write(sop.description)
    else:
        st.write("No active corrective procedures required. Follow scheduled preventive maintenance per manufacturer documentation.")

with tab2:
    col_h, col_p = st.columns(2)
    with col_h:
        st.subheader("⚠️ Safety Hazards & Risk Analysis")
        for h in diag_result.safety_hazards:
            st.warning(h)

        st.subheader("📜 Referenced Technical Standards (Advisory)")
        for s in diag_result.standards_referenced:
            st.write(f"- 🏛️ {s}")

    with col_p:
        st.subheader("🛡️ Recommended Personal Protective Equipment (PPE)")
        for p in diag_result.required_ppe:
            st.write(f"- 🧤 {p}")

        st.subheader("🔧 Candidate Replacement Parts (Compatibility Required)")
        if diag_result.replacement_parts:
            for part in diag_result.replacement_parts:
                st.write(f"- 📦 {part}")
        else:
            st.write("- No replacement part identified from available evidence.")

with tab3:
    st.subheader("🤖 Grounded Field Technician Assistant")
    st.caption("Ask questions about safety clearances, Lockout/Tagout procedures, tool requirements, or troubleshooting.")

    quick_cols = st.columns(3)
    preset_q = ""
    if quick_cols[0].button("⚡ Safe to operate?"):
        preset_q = "Can I turn on this charger for a customer right now?"
    if quick_cols[1].button("🧤 Required PPE?"):
        preset_q = "What PPE gloves and gear are required?"
    if quick_cols[2].button("⏱️ Dispatch SLA?"):
        preset_q = "What is the emergency dispatch SLA for this station?"

    user_q = st.text_input("Your question to AI Assistant:", value=preset_q, placeholder="e.g. Can I touch the cable before LOTO? What is the megohmmeter spec?")

    if st.button("Submit Query", type="primary") and user_q:
        answer = st.session_state.engine.ask(diag_result, user_q)
        st.session_state.chat_history.append((user_q, answer))

    st.markdown("---")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**👨‍🔧 Technician:** {q}")
        st.markdown(f"**🤖 AI Assistant:**\n{a}")
        st.markdown("---")

with tab4:
    st.subheader("📄 AI-Assisted Visual Inspection Work Order Export")
    st.caption("Generate standardized field maintenance work orders for site technicians and management systems.")

    # Generate PDF (Primary)
    pdf_bytes = ReportGenerator.to_pdf(diag_result, image_path=active_image_file_path)
    html_content = ReportGenerator.to_html(diag_result, image_path=active_image_file_path)
    md_content = ReportGenerator.to_markdown(diag_result)

    btn_pdf, btn_html, btn_md = st.columns(3)
    with btn_pdf:
        st.download_button(
            label="📄 Download Work Order PDF (Primary)",
            data=pdf_bytes,
            file_name=f"{diag_result.report_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    with btn_html:
        st.download_button(
            label="🌐 Download Work Order HTML",
            data=html_content,
            file_name=f"{diag_result.report_id}.html",
            mime="text/html",
            use_container_width=True
        )
    with btn_md:
        st.download_button(
            label="📝 Download Work Order Markdown",
            data=md_content,
            file_name=f"{diag_result.report_id}.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### Document Preview (HTML Representation)")
    st.components.v1.html(html_content, height=650, scrolling=True)
