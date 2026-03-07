import tempfile
import cv2
import numpy as np
import streamlit as st
import time
from core.detector import draw_tracked_frame
from core.analyzer import evaluate_room_health

def process_vid(video_file, model, conf, iou, device):

    with st.spinner("Processing Video..."):

        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()

        # Tracking summary variables
        unique_ids = set()
        all_class_counts = {}
        class_conf_lists = {}
        total_conf = []

        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # tracking for IDs (using bytetrack)
            results = model.track(
                frame,
                conf=conf,
                iou=iou,
                device=device,
                persist=True,
                tracker="bytetrack.yaml",
            )

            # appling cvzone boxes
            annotated = draw_tracked_frame(frame, results, model)

            # Collect tracking metrics
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:

                boxes = results[0].boxes

                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    names = model.names

                    for obj_id, cls_id, conf_score in zip(ids, classes, confidences):

                        if obj_id not in unique_ids:
                            unique_ids.add(obj_id)

                            label = names[int(cls_id)]
                            all_class_counts[label] = all_class_counts.get(label, 0) + 1

                            class_conf_lists.setdefault(label, []).append(float(conf_score))

                            total_conf.append(float(conf_score))

            # FPS overlay
            frame_count += 1
            elapsed = time.time() - start_time
            fps_live = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(
                annotated,
                f"FPS: {fps_live:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # Display frame live
            frame_placeholder.image(annotated, channels="BGR")

        cap.release()
        processing_time = time.time() - start_time

        # Per-class confidence stats
        class_details = {}
        for label in all_class_counts:
            confs = np.array(class_conf_lists[label])
            class_details[label] = {
                "count": all_class_counts[label],
                "avg_confidence": round(float(confs.mean()), 3),
                "min_confidence": round(float(confs.min()), 3),
                "max_confidence": round(float(confs.max()), 3),
            }

        room_assessment = evaluate_room_health(all_class_counts)

        summary = {
            "total_detections": len(unique_ids),
            "unique_classes": len(all_class_counts),
            "class_counts": all_class_counts,
            "class_details": class_details,
            "avg_confidence": (
                round(sum(total_conf) / len(total_conf), 3) if total_conf else 0
            ),
            "room_assessment": room_assessment,
            "frames_processed": frame_count,
            "processing_time": round(processing_time, 1),
            "avg_fps": round(fps_live, 1) if frame_count > 0 else 0,
        }

        return summary

def show_results(summary):

    st.success("Detection Completed!")
    st.toast("Detection Completed !", icon="✅")

    show_metrics(summary)

def show_metrics(summary):

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Objects", summary["total_detections"])
    c2.metric("Unique Classes", summary["unique_classes"])
    c3.metric("Avg Confidence", f"{summary['avg_confidence']:.1%}")
    c4.metric("Avg FPS", summary.get("avg_fps", "N/A"))

    # Processing stats
    st.subheader("Processing Info", divider="rainbow")
    pc1, pc2 = st.columns(2)
    pc1.metric("Frames Processed", summary.get("frames_processed", "N/A"))
    pc2.metric("Processing Time", f"{summary.get('processing_time', 'N/A')}s")

    # Per-class equipment breakdown
    st.subheader("Equipment Detected", divider="rainbow")
    class_details = summary.get("class_details", {})
    if class_details:
        for label, details in class_details.items():
            display_name = label.replace("_", " ").title()
            with st.container(border=True):
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Equipment", display_name)
                mc2.metric("Count", details["count"])
                mc3.metric("Avg Conf.", f"{details['avg_confidence']:.1%}")
                mc4.metric(
                    "Range",
                    f"{details['min_confidence']:.0%}-{details['max_confidence']:.0%}",
                )
    else:
        st.info("No objects detected in this video.")

    # Room assessment
    st.subheader("Room Assessment", divider="rainbow")
    room = summary.get("room_assessment", {})
    if room:
        score_text = f"{room['score']} / {room['max_score']} ({room['percentage']}%)"
        st.metric("Room Health Score", score_text)

        if room["level"] == "success":
            st.success(f"Status: {room['status']}")
        elif room["level"] == "warning":
            st.warning(f"Status: {room['status']}")
        else:
            st.error(f"Status: {room['status']}")

        if room["missing_critical"]:
            st.error("Missing Critical Equipment:")
            for item in room["missing_critical"]:
                st.write(f"  - {item}")

        if room["missing_important"]:
            st.warning("Missing Important Equipment:")
            for item in room["missing_important"]:
                st.write(f"  - {item}")

    st.info(
        """• Each object counted once via tracking (unique ID)  
• Confidence reflects first detection per object  
• Room score based on standard equipment checklist  
"""
    )
