import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="FactoryLens AI",
    page_icon="🏭",
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .main-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.1rem;}
    .subtitle {color: #666; margin-bottom: 1.2rem;}
    .status-box {
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Title ----------
st.markdown(
    '<div class="main-title">🏭 FactoryLens AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect → Explain → Analyze → Prevent | Product quality inspection prototype'
    '</div>',
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Inspection Setup")

    product_name = st.text_input(
        "Product Name",
        "Metal Component"
    )

    batch_id = st.text_input(
        "Batch ID",
        "BATCH-001"
    )

    st.divider()

    st.caption("Prototype mode")

    st.caption(
        "The current version uses image characteristics as a demo "
        "AI-style scoring method. A trained computer-vision model "
        "can be connected later."
    )

# ---------- Upload ----------
uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear inspection image of the product."
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    arr = np.asarray(image).astype(np.float32)

    # ---------- Image analysis ----------
    gray = (
        0.299 * arr[:, :, 0]
        + 0.587 * arr[:, :, 1]
        + 0.114 * arr[:, :, 2]
    )

    brightness = float(gray.mean())
    contrast = float(gray.std())

    # Simple texture estimate
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()

    texture = float((gx + gy) / 2)

    # ---------- Demo AI-style scoring ----------
    texture_signal = min(texture / 25.0, 1.0)
    contrast_signal = min(contrast / 90.0, 1.0)

    defect_score = int(
        round(
            (0.65 * texture_signal + 0.35 * contrast_signal) * 100
        )
    )

    # ---------- Prediction ----------
    if defect_score >= 65:

        status = "DEFECT DETECTED"
        defect_type = "Possible surface crack / irregularity"
        severity = "High"

        batch_quality = max(
            45,
            100 - defect_score
        )

        recommendation = (
            "Hold the item for manual inspection and check nearby "
            "units from the same batch."
        )

        cause = (
            "Possible surface stress, handling damage, machining "
            "variation, or material irregularity."
        )

    elif defect_score >= 40:

        status = "REVIEW REQUIRED"
        defect_type = "Possible surface anomaly"
        severity = "Medium"

        batch_quality = max(
            60,
            100 - defect_score // 2
        )

        recommendation = (
            "Send for secondary inspection before accepting the item."
        )

        cause = (
            "Possible lighting variation, surface texture, contamination, "
            "or minor manufacturing variation."
        )

    else:

        status = "NO OBVIOUS DEFECT"
        defect_type = "No obvious surface anomaly"
        severity = "Low"

        batch_quality = min(
            99,
            100 - defect_score // 2
        )

        recommendation = (
            "Accept provisionally and continue normal quality checks."
        )

        cause = (
            "No strong defect signal found in the uploaded image."
        )

    confidence = int(
        min(
            99,
            max(
                55,
                55 + abs(defect_score - 50) * 0.8
            )
        )
    )

    # ---------- Metrics ----------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Inspection Status",
        status
    )

    c2.metric(
        "Confidence",
        f"{confidence}%"
    )

    c3.metric(
        "Severity",
        severity
    )

    c4.metric(
        "Batch Quality",
        f"{batch_quality}%"
    )

    st.divider()

    # ---------- Image + explanation ----------
    left, right = st.columns([1.25, 1])

    with left:

        st.subheader("Inspection Image")

        st.image(
            image,
            use_container_width=True
        )

        # Prototype inspection-region highlighting
        local_gx = np.abs(np.diff(gray, axis=1))
        local_gy = np.abs(np.diff(gray, axis=0))

        grad = np.zeros_like(gray)

        grad[:, 1:] += local_gx
        grad[1:, :] += local_gy

        h, w = grad.shape

        y0 = int(h * 0.15)
        y1 = int(h * 0.85)

        x0 = int(w * 0.15)
        x1 = int(w * 0.85)

        region = grad[y0:y1, x0:x1]

        annotated = image.copy()

        draw = ImageDraw.Draw(annotated)

        if region.size:

            peak_y, peak_x = np.unravel_index(
                np.argmax(region),
                region.shape
            )

            cx = x0 + peak_x
            cy = y0 + peak_y

            box_w = max(
                50,
                int(w * 0.22)
            )

            box_h = max(
                50,
                int(h * 0.18)
            )

            bx0 = max(
                0,
                cx - box_w // 2
            )

            by0 = max(
                0,
                cy - box_h // 2
            )

            bx1 = min(
                w - 1,
                cx + box_w // 2
            )

            by1 = min(
                h - 1,
                cy + box_h // 2
            )

            draw.rectangle(
                [
                    bx0,
                    by0,
                    bx1,
                    by1
                ],
                outline="red",
                width=5
            )

            st.subheader(
                "Potential Inspection Region"
            )

            st.image(
                annotated,
                caption=(
                    "Prototype highlight based on image texture. "
                    "It is not a trained object-detection model."
                ),
                use_container_width=True
            )

    with right:

        st.subheader("AI Prediction")

        st.markdown(
            f"""
            <div class="status-box">
                <b>{status}</b><br>
                Predicted defect type:
                <b>{defect_type}</b><br>
                Confidence:
                <b>{confidence}%</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("**Possible Cause**")

        st.write(cause)

        st.write("**Recommended Action**")

        st.write(recommendation)

        st.write("**Quality Score**")

        st.progress(
            batch_quality / 100
        )

        st.write("**Image Signals**")

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "Brightness",
            f"{brightness:.1f}"
        )

        s2.metric(
            "Contrast",
            f"{contrast:.1f}"
        )

        s3.metric(
            "Texture",
            f"{texture:.1f}"
        )

    # ---------- Batch analysis ----------
    st.divider()

    st.subheader("📊 Batch Analysis")

    history_row = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Product": product_name,
        "Batch": batch_id,
        "Status": status,
        "Defect Type": defect_type,
        "Severity": severity,
        "Confidence": confidence,
        "Quality": batch_quality,
    }

    if (
        not st.session_state.history
        or st.session_state.history[-1] != history_row
    ):
        st.session_state.history.append(
            history_row
        )

    hist_df = pd.DataFrame(
        st.session_state.history
    )

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "Items Inspected",
        len(hist_df)
    )

    b2.metric(
        "Review / Defect Cases",
        int(
            (
                hist_df["Status"]
                != "NO OBVIOUS DEFECT"
            ).sum()
        )
    )

    b3.metric(
        "Average Quality",
        f"{hist_df['Quality'].mean():.1f}%"
    )

    st.dataframe(
        hist_df,
        use_container_width=True,
        hide_index=True
    )

    # ---------- Human feedback ----------
    st.subheader("👤 Human Verification")

    st.caption(
        "Use this to capture operator feedback. "
        "This is a prototype feedback workflow."
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        if st.button(
            "✅ Prediction Correct",
            use_container_width=True
        ):
            st.success(
                "Feedback recorded: prediction correct."
            )

    with f2:

        if st.button(
            "❌ Prediction Incorrect",
            use_container_width=True
        ):
            st.warning(
                "Feedback recorded: prediction needs review."
            )

    with f3:

        if st.button(
            "🔍 Send for Manual Review",
            use_container_width=True
        ):
            st.info(
                "Item marked for manual review."
            )

    # ---------- Download report ----------
    report = f"""
FACTORYLENS AI - INSPECTION REPORT
------------------------------------

Date/Time:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Product:
{product_name}

Batch ID:
{batch_id}

Image:
{uploaded_file.name}

Inspection Status:
{status}

Defect Type:
{defect_type}

Confidence:
{confidence}%

Severity:
{severity}

Quality Score:
{batch_quality}%

Possible Cause:
{cause}

Recommended Action:
{recommendation}

NOTE:
This is a prototype. The prediction currently uses image
characteristics as a demonstration and is not a trained
industrial defect-detection model.
"""

    st.download_button(
        "⬇️ Download Inspection Report",
        data=report,
        file_name=f"{batch_id}_inspection_report.txt",
        mime="text/plain",
        use_container_width=True
    )

else:

    st.info(
        "👆 Upload a product image above to start the inspection."
    )

    st.subheader(
        "Why FactoryLens AI is different"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.write("📷 **Detect**")

        st.write(
            "Inspect an uploaded product image."
        )

    with b:

        st.write("🧠 **Explain**")

        st.write(
            "Show defect type, confidence, severity and possible cause."
        )

    with c:

        st.write("📊 **Analyze**")

        st.write(
            "Track inspection history and batch quality."
        )

    with d:

        st.write("🛠️ **Prevent**")

        st.write(
            "Recommend the next quality-control action."
        )

    st.warning(
        "Prototype note: connect a trained computer-vision "
        "model later for real defect detection."
    )