"""Streamlit user interface for pedestrian detection and distance estimation."""

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from src.detector import PedestrianDetector
from src.video_processor import process_video

st.set_page_config(page_title="Pedestrian Vision Lab", page_icon="P", layout="wide", initial_sidebar_state="expanded")

MODEL_OPTIONS = {
    "YOLOv8s - better accuracy": "yolov8s.pt",
    "YOLOv8n - fastest": "yolov8n.pt",
}

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "last_upload_name" not in st.session_state:
    st.session_state.last_upload_name = None

st.markdown("""
<style>
.block-container { max-width: 1160px; padding-top: 1.5rem; padding-bottom: 3rem; }
.app-title { margin: .2rem 0 .25rem; font-size: 2rem; font-weight: 800; letter-spacing: 0; }
.app-subtitle { margin: 0 0 1rem; color: #8b949e; font-size: 1rem; }
.section-title { font-size: 1.15rem; font-weight: 750; margin: 1rem 0 .25rem; }
.muted { color: #8b949e; font-size: .9rem; }
.status-row { display: flex; gap: .7rem; flex-wrap: wrap; margin: .8rem 0 1rem; }
.status-pill { border: 1px solid rgba(139, 148, 158, .35); border-radius: 8px; padding: .55rem .7rem; color: inherit; background: rgba(139, 148, 158, .08); }
.status-pill strong { display: block; font-size: .8rem; }
.status-pill span { color: #8b949e; font-size: .76rem; }
.preview-frame { border: 1px solid rgba(139, 148, 158, .35); border-radius: 8px; overflow: hidden; background: rgba(139, 148, 158, .08); }
div[data-testid="stMetric"] {
    background: rgba(139, 148, 158, .08);
    border: 1px solid rgba(139, 148, 158, .35);
    padding: .8rem;
    border-radius: 8px;
}
div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] { color: inherit; }
div[data-testid="stDownloadButton"] button, div.stButton > button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 class="app-title">Pedestrian Detection Dashboard</h1>
<p class="app-subtitle">Upload a traffic video, run a pretrained YOLOv8 person detector, review a reliable annotated frame, and download the complete marked video.</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="status-row">
  <div class="status-pill"><strong>1. Upload</strong><span>MP4, AVI, MOV, MKV</span></div>
  <div class="status-pill"><strong>2. Analyze</strong><span>Selectable YOLOv8 model</span></div>
  <div class="status-pill"><strong>3. Review</strong><span>Preview frame and metrics</span></div>
  <div class="status-pill"><strong>4. Export</strong><span>Download annotated MP4</span></div>
</div>
""", unsafe_allow_html=True)


def read_preview_frame(video_path: Path) -> bytes | None:
    """Return a JPEG preview from the middle of the processed video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 1:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return encoded.tobytes()

with st.sidebar:
    st.markdown("## Analysis settings")
    st.caption("Adjust these values before processing. They affect displayed detections and distance.")
    model_label = st.selectbox(
        "Detection model",
        list(MODEL_OPTIONS.keys()),
        help="YOLOv8s is the recommended balance. YOLOv8n is faster on slow laptops.",
    )
    confidence = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
    real_height = st.number_input("Assumed pedestrian height (m)", 0.5, 2.5, 1.7, 0.1)
    focal_length = st.number_input("Camera focal length (px)", 100.0, 3000.0, 700.0, 25.0)
    st.divider()
    st.markdown("**Distance model**")
    st.latex(r"D = \frac{H \times F}{h}")
    st.caption("Approximate estimate, not ground-truth depth. Calibrate focal length for better results.")

st.markdown('<div class="section-title">Upload a dashcam video</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Supported formats: MP4, AVI, MOV, MKV · YOLOv8s is selected by default for better detection quality.</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop a video here or browse your files", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")

if uploaded_file:
    original_bytes = uploaded_file.getvalue()
    if st.session_state.last_upload_name != uploaded_file.name:
        st.session_state.analysis_result = None
        st.session_state.last_upload_name = uploaded_file.name

    st.success(f"Ready: **{uploaded_file.name}** ({len(original_bytes) / 1024 / 1024:.1f} MB)")
    top_left, top_right = st.columns([1, 1], gap="large")
    with top_left:
        st.markdown("### Input video")
        st.video(original_bytes)
    with top_right:
        st.markdown("### Output preview")
        if st.session_state.analysis_result and st.session_state.analysis_result["preview_image"]:
            st.image(st.session_state.analysis_result["preview_image"], caption="Annotated frame from processed video", use_container_width=True)
        else:
            st.info("Run analysis to generate the annotated preview and output file.")

    process_clicked = st.button("Run pedestrian analysis", type="primary", use_container_width=True)
    if process_clicked:
        progress = st.progress(0, text="Preparing analysis...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                input_path = temp_dir / uploaded_file.name
                output_path = temp_dir / "annotated_output.mp4"
                input_path.write_bytes(original_bytes)
                model_path = MODEL_OPTIONS[model_label]
                progress.progress(0.02, text=f"Loading {model_path}...")
                detector = PedestrianDetector(model_path=model_path, confidence=confidence)
                stats = process_video(input_path, output_path, detector, real_height, focal_length, lambda value: progress.progress(max(0.02, value), text="Analyzing video frames..."))
                output_bytes = output_path.read_bytes()
                preview_image = read_preview_frame(output_path)
                st.session_state.analysis_result = {
                    "output_bytes": output_bytes,
                    "preview_image": preview_image,
                    "stats": stats,
                    "model_label": model_label,
                    "confidence": confidence,
                    "real_height": real_height,
                    "focal_length": focal_length,
                }
            progress.progress(1.0, text="Analysis complete")
        except Exception as exc:
            progress.empty()
            st.error(f"Analysis failed: {exc}")

    if st.session_state.analysis_result:
        output_bytes = st.session_state.analysis_result["output_bytes"]
        preview_image = st.session_state.analysis_result["preview_image"]
        stats = st.session_state.analysis_result["stats"]
        st.markdown("### Result")
        result_left, result_right = st.columns([2, 1], gap="large")
        with result_left:
            if preview_image:
                st.image(preview_image, caption="Annotated preview frame", use_container_width=True)
            else:
                st.info("Preview frame could not be created, but the output video is available below.")
        with result_right:
            st.download_button("Download annotated MP4", output_bytes, "annotated_output.mp4", "video/mp4", use_container_width=True)
            with st.expander("Open video player"):
                st.video(output_bytes)
            st.caption(f"Model: {st.session_state.analysis_result['model_label']}")
            st.caption("If the browser player appears black, use the download. The downloaded file contains the full annotated output.")

        if stats.total_detections == 0:
            st.warning(
                "No pedestrians were detected in this video. Try lowering the confidence slider, "
                "using a clearer pedestrian video, or choosing frames where people are larger and less blurred."
            )

        st.markdown("### Analysis summary")
        metric_cols = st.columns(3)
        values = [("Frames", stats.total_frames), ("Detections", stats.total_detections), ("Max per frame", stats.max_pedestrians)]
        for column, (label, value) in zip(metric_cols, values):
            column.metric(label, value)
        metric_cols = st.columns(3)
        values = [("Average per frame", f"{stats.average_detections_per_frame:.2f}"), ("Processing time", f"{stats.processing_seconds:.2f}s"), ("Processing FPS", f"{stats.processing_fps:.2f}")]
        for column, (label, value) in zip(metric_cols, values):
            column.metric(label, value)
else:
    st.session_state.analysis_result = None
    st.session_state.last_upload_name = None
    st.info("Upload a dashcam video to begin. For local testing, place one at `sample/dashcam.mp4` and upload it here.")

with st.expander("About the distance estimate"):
    st.write("The app estimates distance from bounding-box height using the pinhole-camera relationship. Accuracy depends on focal-length calibration, actual pedestrian height, camera angle, perspective, lens characteristics, and bounding-box quality.")
    st.caption("Recommended calibration: F = (pixel height × known distance) / known height")

st.divider()
st.caption("YOLOv8 selectable model · COCO person class · OpenCV video processing · Streamlit interface")
