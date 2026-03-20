import streamlit as st
from PIL import Image
import numpy as np
import torch
from core.detector import detect, draw_detections
from core.analyzer import summarize
from core.model import load_model
import base64, time
from core.Video_process import process_vid, show_results


def load_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

base64_logo = load_logo("assets/logo.png")

st.set_page_config("MediScan AI", layout="wide", page_icon="assets/favicon.png")
st.logo("assets/logo.png", size="large")

# global session states
if "output_path" not in st.session_state:
    st.session_state.output_path = "camera_output.mp4"

if "recording" not in st.session_state:
    st.session_state.recording = False

# loading custom css
with open("core/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# main title with logo
st.markdown(
    f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{base64_logo}" width="180">
    </div>
    """,
    unsafe_allow_html=True,
)

# SIDE BAR CONTROLS
with st.sidebar:
    st.markdown(
        "<div class='main-title'> <span class='rainbow-text'> Menu </span> </div>",
        unsafe_allow_html=True,
    )
    st.subheader("", divider="rainbow")
    page = st.segmented_control(
        "Navigate to:", ["Home", "Detection", "About"], default="Home"
    )

    # getting model
    selected_model = st.selectbox("Select Model:", ["Nano", "Small", "Medium"], index=0)

    # loading models
    if selected_model == "Nano":
        model = load_model("Nano")

    elif selected_model == "Small":
        model = load_model("Small")
        st.toast("Model Switched to Small", duration="short")

    elif selected_model == "Medium":
        model = load_model("Medium")
        st.toast("Model Switched to Medium", duration="short")

    conf = st.slider(
        "Confidence Range", min_value=0.0, max_value=1.0, step=0.01, value=0.5
    )
    iou = st.slider("IOU Range", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    device = st.selectbox("Device", ["CPU", "GPU"], index=0)
    if device == "GPU":
        # checking if GPU is available
        try:
            if torch.cuda.is_available():
                device = "cuda"
                st.toast("Switched to GPU", duration="short")
            else:
                st.error("GPU not available on this machine. Falling back to CPU.")
                device = "cpu"
        except ImportError:
            st.error("PyTorch not found. Falling back to CPU.")
            device = "cpu"
    else:
        device = "cpu"

    st.subheader("", divider="rainbow")
    st.info(
        "This is a Medical Equipments Detection app, which uses real time detection model (YOLO). This is a demo app only !",
        icon="⚠️",
    )

if page == "Home":

    # home heading
    st.markdown(
        "<div class='main-title'> <span class='rainbow-text'> MediScan AI </span> </div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>AI-Powered Medical Equipment Detection</div>", unsafe_allow_html=True)
    st.subheader("", divider="rainbow")

    st.markdown("<div class='infoheading'> What is <span class='rainbow-text'>Mediscan AI </span>?</div>", unsafe_allow_html=True)

    st.markdown(
        """<div class='infotxt'>
    MediScan AI is an intelligent object detection system designed to identify
    medical equipment from images and videos using deep learning.            
    </div>""",
    unsafe_allow_html=True
    )

    st.markdown(
        """
    ### Key <span class='rainbow-text'> Features </span>

    - Detect medical equipment in images
    - Analyze recorded videos
    - Adjustable confidence & IOU thresholds
    - GPU acceleration support
    - Clean and structured results summary
    """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
    ### How It <span class='words'> Works </span>

    1. Select a model from the sidebar
    2. Upload an image or video
    3. Adjust confidence & IOU settings
    4. Run detection and view results
    """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
    ### Why This <span class='words'> Matters </span>

    Automated detection can assist in:
    - Equipment inventory monitoring
    - Smart hospital systems
    - Safety compliance checks
    - AI-assisted healthcare environments
    """,
        unsafe_allow_html=True
    )

if page == "Detection":

    # detection heading
    st.markdown("<div class='main-title'> <span class='rainbow-text'> Medical Equipment Detection </span> </div>",unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Detection Page to Detect Medical Objects</div>", unsafe_allow_html=True)
    st.subheader("", divider="rainbow")

    st.markdown("<div style='font-size: 20px;' > Select Detection Method: </div>", unsafe_allow_html=True)

    D_choice = st.segmented_control(
        "",
        ["Image", "Video"],
        default="Image",
    )

    if D_choice == "Image":

        # uploading multiple images
        uploaded_files = st.file_uploader(
            "Upload Image", type=["jpg", "png", "jpeg"], accept_multiple_files=True  # Set False, if single file only
        )

        # button position logic
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            btn = st.button("Detect", help="Click to Detect", use_container_width=True)

            if btn and uploaded_files:
                with st.spinner("Detecting..."):
                    time.sleep(1)
                    st.toast("Detection Completed !", icon="✅")
                    time.sleep(1)

            elif btn and not uploaded_files:
                st.error("Please upload an image !!")

        # gathering all results here
        all_results = []

        # showing detections
        with st.expander("See Detections", expanded=True):
            col1, col2 = st.columns([1.5, 1], border=True, gap="medium")

            with col1:
                with st.expander("Detections", expanded=True):
                    if uploaded_files and btn:
                        for uploaded in uploaded_files:
                            image = np.array(Image.open(uploaded).convert("RGB"))

                            results = detect(model, image, conf, iou, device)

                            all_results.extend(results)

                            annotated = draw_detections(image, results)
                            
                            # showing detected image
                            st.image(annotated, channels="BGR")
            with col2:
                with st.expander("Summary", expanded=True):
                    st.subheader("Detection Summary")
                    st.subheader("", divider="rainbow")
                    if all_results:

                        # summary logic
                        summary = summarize(all_results, model)

                        # metrics
                        m1, m2 = st.columns(2)
                        m1.metric("Total Detections", summary["total_detections"])
                        m2.metric("Unique Classes", summary["unique_classes"])

                        m3, m4 = st.columns(2)
                        m3.metric("Avg Confidence", f"{summary['avg_confidence']:.1%}") # in percentage with 1 last decimal
                        m4.metric(
                            "Conf. Range",
                            f"{summary['min_confidence']:.0%} - {summary['max_confidence']:.0%}",
                        )

                        # class breakdown
                        st.subheader("Equipment Detected", divider="rainbow")
                        class_details = summary.get("class_details", {})
                        if class_details:
                            for label, details in class_details.items():
                                display_name = label.replace("_", " ").title()
                                with st.container(border=True):
                                    dc1, dc2 = st.columns(2)
                                    dc1.metric(display_name, f"x{details['count']}")
                                    dc2.metric(
                                        "Confidence", f"{details['avg_confidence']:.1%}"
                                    )
                        else:
                            st.info("No objects detected.")

                        # showing Low confidence alerts
                        alerts = summary.get("low_confidence_alerts", [])
                        if alerts:
                            st.subheader("Low Confidence Alerts", divider="rainbow")
                            for alert in alerts:
                                st.warning(
                                    f"{alert['class'].replace('_', ' ').title()} \u2014 {alert['confidence']:.1%} confidence"
                                )
                    else:
                        st.info("Please Upload Image and Click Detect")
        
        # for user 
        with st.expander("📘 Instructions", expanded=False):
            st.markdown(
                """
            #### 1️⃣ Select Detection Type
            Choose one of the following:
            - **Image** → Upload one or multiple images.
            - **Video** → Upload a recorded video file (.mp4).

            #### 2️⃣ Select Model
            - **Nano** → Faster performance, lightweight.
            - **Small** → More accurate, slightly slower.
            - **Medium** → Balanced performance and accuracy <span class='new-badge'>New</span>

            #### 3️⃣ Adjust Detection Parameters
            - **Confidence Threshold**  
            Controls how confident the model must be before detecting an object.  
            Higher value = fewer but more certain detections.
            
            - **IOU Threshold**  
            Controls overlap filtering between bounding boxes.  
            Higher value = stricter duplicate filtering.

            #### 4️⃣ Choose Device
            - **CPU** → Standard processing.
            - **GPU** → Faster detection (if available).

            #### 5️⃣ Run Detection
            Click the **Detect** button to process the image or video.

            ---

            ### 📊 Output Includes
            - Annotated image/video with bounding boxes
            - Detection summary
            - Total objects detected
            - Average confidence score
            - Class-wise object breakdown

            ---

            ### ⚠️ Important Notes
            - Works only on trained medical equipment classes.
            - Performance depends on image quality and lighting.
            - Higher resolution videos may take longer to process.
            """
            )

    if D_choice == "Video":
        
        # uploading video
        video_files = st.file_uploader(
            "Upload Video", type=["mp4"], accept_multiple_files=True)  # multiple files, now this is a list
        
        if video_files:
            for video_file in video_files:  # getting only 1 element of list and processing it

                original = video_file

                # btn position logic
                c1, c2, c3 = st.columns([2, 1, 2])
                with c2:
                    viddetect_btn = st.button(
                        "Detect", help="Click to Detect", use_container_width=True
                    )

                if viddetect_btn:
                    with st.expander("Detection Results:", expanded=True):
                        summary = process_vid(video_file, model, conf, iou, device)
                        show_results(summary)

        with st.expander("📘 Instructions", expanded=False):
            st.markdown(
                """
            #### 1️⃣ Select Detection Type
            Choose one of the following:
            - **Image** → Upload one or multiple images.
            - **Video** → Upload a recorded video file (.mp4).

            #### 2️⃣ Select Model
            - **Nano** → Faster performance, lightweight.
            - **Small** → More accurate, slightly slower.
            - **Medium** → Balanced performance and accuracy <span class='new-badge'>New</span>

            #### 3️⃣ Adjust Detection Parameters
            - **Confidence Threshold**  
            Controls how confident the model must be before detecting an object.  
            Higher value = fewer but more certain detections.
            
            - **IOU Threshold**  
            Controls overlap filtering between bounding boxes.  
            Higher value = stricter duplicate filtering.

            #### 4️⃣ Choose Device
            - **CPU** → Standard processing.
            - **GPU** → Faster detection (if available).

            #### 5️⃣ Run Detection
            Click the **Detect** button to process the image or video.

            ---

            ### 📊 Output Includes
            - Annotated image/video with bounding boxes
            - Detection summary
            - Total objects detected
            - Average confidence score
            - Class-wise object breakdown

            ---

            ### ⚠️ Important Notes
            - Works only on trained medical equipment classes.
            - Performance depends on image quality and lighting.
            - Higher resolution videos may take longer to process.
            """
            )

if page == "About":

    # about heading
    st.markdown(
        "<div class='main-title'>About <span class='rainbow-text'> MediScan AI </span></div>",
        unsafe_allow_html=True,
    )
    st.subheader("", divider="rainbow")

    st.markdown(
        """
    ### <span class='rainbow-text'> Technology </span> Used

    MediScan AI is built using:

    - Python
    - Streamlit (Frontend Interface)
    - OpenCV (Image Processing)
    - YOLO (Ultralytics) for Object Detection
    - PyTorch (Deep Learning Backend)
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    ### Model <span class='rainbow-text'>Architecture </span>

    The system uses YOLO (You Only Look Once), a real-time object detection model
    capable of detecting multiple objects in a single forward pass.

    Two model sizes are supported:
    - <span class='rainbow-text'>Nano</span> (faster, lightweight)
    - <span class='rainbow-text'>Small</span> (more accurate)
    - <span class='rainbow-text'>Medium</span> (balanced) <span class='new-badge'>New</span>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    ### Application in <span class='rainbow-text'> Healthcare </span>

    This system demonstrates how AI can assist in:

    - Monitoring medical equipment
    - Detecting missing or misplaced items
    - Smart hospital automation systems
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    ### Current <span class='words'>Limitations</span>

    - Works only on trained equipment classes
    - Performance depends on lighting and image quality
    - Not certified for clinical deployment
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    ### Future <span class='rainbow-text'>Improvements</span>

    - Real-time camera monitoring
    - Database integration
    - Equipment tracking analytics
    - Multi-class hospital safety monitoring
    """,
        unsafe_allow_html=True,
    )

    st.subheader("", divider="rainbow")

    # love button lol 
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button(
            "❤️ Click To Show LOVE ❤️",
            help="Show Love, Support Us!",
            use_container_width=True,
        ):
            st.balloons()
            st.toast("Thanks for Supporting US !", icon="❤️")
