import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Ensure the repo root is on sys.path so "apps.*" imports work in Streamlit.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.infer import load_model, predict_risk, predict_risk_with_cam


def _format_risk(risk: float) -> str:
    """Return risk as a percentage string with one decimal place."""
    return f"{risk * 100:.1f}%"


def _interpret_risk(risk: float) -> str:
    """Map risk probability to a human-friendly bucket."""
    percent = risk * 100
    if percent < 20:
        return "Low Risk"
    if percent <= 50:
        return "Moderate Risk"
    return "High Risk"


def main() -> None:
    st.set_page_config(page_title="Mole Risk Analyzer (Demo)")
    st.title("Mole Risk Analyzer (Demo)")

    st.info("This tool is for educational purposes only and is NOT medical advice.")

    # Load the model once per session and surface any errors early.
    try:
        load_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a JPG or PNG image", type=["jpg", "jpeg", "png"]
    )

    show_cam = st.checkbox("Show Grad-CAM heatmap (explainability)")
    alpha = None
    if show_cam:
        alpha = st.slider("Heatmap opacity (alpha)", 0.1, 0.8, 0.45, 0.05)

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", width="stretch")
        except Exception as exc:
            st.error(f"Could not read the uploaded image: {exc}")
            st.stop()
    else:
        image = None

    if st.button("Estimate Risk"):
        if image is None:
            st.warning("Please upload an image before estimating risk.")
        else:
            try:
                if show_cam:
                    risk, overlay = predict_risk_with_cam(image, alpha=alpha or 0.45)
                    st.success(
                        f"Melanoma risk: {_format_risk(risk)} ({_interpret_risk(risk)})"
                    )
                    st.image(overlay, caption="Grad-CAM overlay", width="stretch")
                    st.caption(
                        "Grad-CAM highlights image regions that most influenced the model output. "
                        "It does NOT prove clinical features."
                    )
                else:
                    risk = predict_risk(image)
                    st.success(
                        f"Melanoma risk: {_format_risk(risk)} ({_interpret_risk(risk)})"
                    )

                st.caption(
                    "Interpretation: <20% = Low Risk, 20-50% = Moderate Risk, >50% = High Risk."
                )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")


if __name__ == "__main__":
    main()
