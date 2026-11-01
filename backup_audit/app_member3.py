"""
Streamlit Web Application for Member 3:
Diagnostic Assistant, Safety Severity Scoring & Maintenance Recommendations.

Features:
  - Direct integration with Member 2's MobileNetV3 Dual-Head Vision Model.
  - Image upload or interactive sample dataset browser.
  - Contextual weather escalation (Rain escalates coupler defects to CRITICAL).
  - Step-by-step SOPs aligned with IEC 61851-23, IEC 62196-3, and NFPA 70E.
  - Mandatory PPE and spare parts bill of materials.
  - Interactive Grounded Technician Q&A Assistant.
  - Certified Work Order download (HTML and Markdown).
"""

import sys
from pathlib import Path
from PIL import Image
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from src.diagnostics import (
    EVDiagnosticEngine,
    ChargerContext,
    ReportGenerator,
    VisualInspectionStatus,
    SeverityLevel,
)

# Page configuration
st.set_page_config(
    page_title="EV Fault Diagnostic Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "engine" not in st.session_state:
    st.session_state.engine = EVDiagnosticEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title & Header
st.title("⚡ Vision-Language AI Assistant for EV Charging Stations")
st.subheader("Intelligent Fault Diagnosis, Severity Scoring & SOP Engine")
st.caption("Neuro-Symbolic Diagnostic Architecture: Deterministic Safety Rules (IEC / NFPA) + Grounded Multimodal Assistant")
st.markdown("---")

# ------------------------------------------------------------------
# Sidebar: Configuration & Image Source
# ------------------------------------------------------------------
st.sidebar.header("🛠️ Station Operating Context")
station_id = st.sidebar.text_input("Station Identifier", value="EVCS-IND-PUNE-01")

charger_type = st.sidebar.selectbox(
    "Hardware Architecture",
    [
        "150kW DC Fast Charger (CCS2 Dual Gun)",
        "350kW Ultra-Fast Liquid-Cooled Dispenser (CCS2)",
        "50kW DC Fast Charger (CCS2 / CHAdeMO)",
        "22kW AC Commercial Dispenser (Type 2)"
    ]
)

weather = st.sidebar.selectbox(
    "Ambient Weather Condition",
    [
        "Clear / Dry",
        "Monsoon Rain / Wet Ground",
        "High Humidity / Fog",
        "Extreme Heat (>45°C)"
    ]
)

has_ocpp = st.sidebar.checkbox("Simulate Live OCPP 2.0.1 Telemetry Link", value=False)
ocpp_raw = "Operative" if has_ocpp else None

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
    "Dented Cabinet Housing (charger / broken)": dataset_base / "charger" / "broken" / "charger_broken_0000.jpg",
    "Healthy Charging Cable (cord / healthy)": dataset_base / "cord" / "healthy" / "cord_healthy_0094.jpg",
    "Healthy Connector Plug (plug / healthy)": dataset_base / "plug" / "healthy" / "plug_healthy_0130.jpg",
    "Healthy Display Screen (screen / healthy)": dataset_base / "screen" / "healthy" / "screen_healthy_0174.jpg",
    "Healthy Charger Kiosk (charger / healthy)": dataset_base / "charger" / "healthy" / "charger_healthy_0023.jpg",
}

active_image = None
active_image_desc = ""

if source_mode == "Test Dataset Library":
    selected_sample = st.sidebar.selectbox("Choose Verified Sample Image", list(sample_images.keys()))
    img_path = sample_images[selected_sample]
    if img_path.exists():
        active_image = Image.open(img_path).convert("RGB")
        active_image_desc = str(img_path)
    else:
        st.sidebar.error(f"Sample file not found: {img_path}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload station photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file).convert("RGB")
        active_image_desc = uploaded_file.name

threshold = st.sidebar.slider("Broken Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

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
    ocpp_state_raw=ocpp_raw
)

# Run Inference & Diagnostic Evaluation
with st.spinner("Running MobileNetV3-Small Dual-Head CV Inference & Diagnostic Engine..."):
    diag_result = st.session_state.engine.diagnose_image(
        active_image,
        context=ctx,
        station_id=station_id,
        threshold=threshold
    )

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

    st.write(f"**Condition Confidence ({diag_result.condition}):** {diag_result.confidence_score*100:.1f}%")
    st.progress(float(diag_result.confidence_score))

    comp_conf = float(meta.get("component_confidence", 0.0))
    st.write(f"**Component Confidence ({diag_result.component}):** {comp_conf*100:.1f}%")
    st.progress(comp_conf)

    if diag_result.condition == "broken" and diag_result.confidence_score < 0.65:
        st.warning("⚠️ Low AI confidence (< 65%). On-site technician verification strongly advised.")

# ------------------------------------------------------------------
# Status & Severity HUD Banner
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

st.markdown(
    f"""
    <div style="background-color: {bg_color}; border-left: 6px solid {text_color}; padding: 14px 20px; border-radius: 4px; margin: 18px 0;">
        <h3 style="color: {text_color}; margin: 0 0 6px 0;">{icon} {diag_result.visual_inspection_status.value}{weather_text}</h3>
        <span style="color: #444;">Rationale: <strong>{diag_result.root_cause_analysis}</strong></span>
    </div>
    """,
    unsafe_allow_html=True
)

# Metric Badges
m1, m2, m3, m4 = st.columns(4)
m1.metric("Visual Inspection Status", diag_result.visual_inspection_status.value)
m2.metric("Safety Severity Level", diag_result.overall_severity.value)
m3.metric("Maintenance Priority", diag_result.maintenance_priority.value)
m4.metric("Dispatch SLA", f"{diag_result.dispatch_sla_hours} Hours" if diag_result.dispatch_sla_hours else "Routine")

# ------------------------------------------------------------------
# Diagnostic Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Standard Operating Procedures (SOP)",
    "🛡️ Safety Hazards & PPE",
    "🤖 Field Technician Assistant (Q&A)",
    "📄 Work Order Ticket Export"
])

with tab1:
    st.subheader("Observable Findings & Observable Evidence")
    for exp in diag_result.fault_explanations:
        st.info(exp)

    st.subheader("Step-by-Step Maintenance Directives (IEC / NFPA)")
    if diag_result.recommended_sops:
        for sop in diag_result.recommended_sops:
            badge = "🚨 **[SAFETY CRITICAL]**" if sop.is_safety_critical else f"ℹ️ [{sop.action_type.upper()}]"
            with st.expander(f"Step {sop.step_number}: {sop.title} {badge}", expanded=sop.is_safety_critical):
                st.write(sop.description)
    else:
        st.write("No active repair procedures required. Follow scheduled preventive maintenance.")

with tab2:
    col_h, col_p = st.columns(2)
    with col_h:
        st.subheader("⚠️ Safety Hazards & Compliance Directives")
        for h in diag_result.safety_hazards:
            st.warning(h)

        st.subheader("📜 Referenced Standards")
        for s in diag_result.standards_referenced:
            st.write(f"- 🏛️ {s}")

    with col_p:
        st.subheader("🛡️ Mandatory Personal Protective Equipment (PPE)")
        for p in diag_result.required_ppe:
            st.write(f"- 🧤 {p}")

        st.subheader("🔧 Candidate Replacement Parts (BOM)")
        if diag_result.replacement_parts:
            for part in diag_result.replacement_parts:
                st.write(f"- 📦 {part}")
        else:
            st.write("- None currently required.")

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
    st.subheader("📄 Certified Maintenance Work Order Ticket")
    st.caption("Generate official work order documentation for field service dispatch.")

    md_content = ReportGenerator.to_markdown(diag_result)
    html_content = ReportGenerator.to_html(diag_result)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.download_button(
            label="⬇️ Download Work Order (HTML Ticket)",
            data=html_content,
            file_name=f"WorkOrder_{station_id}.html",
            mime="text/html",
            use_container_width=True
        )
    with btn_col2:
        st.download_button(
            label="⬇️ Download Work Order (Markdown)",
            data=md_content,
            file_name=f"WorkOrder_{station_id}.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.markdown("#### Document Preview (Markdown)")
    st.code(md_content, language="markdown")
