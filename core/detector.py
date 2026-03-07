# core/detector.py

import cv2
import cvzone # for more better boxes

# colors for each class (BGR)
CLASS_COLORS = {
    "monitor": (255, 140, 50),
    "hospital_bed": (50, 190, 50),
    "infusion_pole": (50, 170, 255),
    "stretcher": (50, 210, 210),
    "wheelchair": (210, 190, 50),
}
DEFAULT_COLOR = (180, 180, 180)


def detect(model, image, conf, iou, device):
    results = model(image, conf=conf, iou=iou, device=device, imgsz=640)
    return results


def draw_detections(image, results):
    
    annotated = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    if results is None or len(results) == 0:
        return annotated

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return annotated

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    names = r.names

    for box, cls_id, conf_val in zip(xyxy, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        label = names[int(cls_id)]
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(
            annotated, (x1, y1, w, h), l=30, t=4, rt=1, colorR=color, colorC=[255, 0, 255]
        )

        text = f"{label.replace('_', ' ').title()} {conf_val:.0%}"

        cvzone.putTextRect(
            annotated,
            text,
            (max(0, x1), max(30, y1 - 10)),
            scale=0.8,
            thickness=1,
            colorT=(255, 255, 255),
            colorR=color,
            font=cv2.FONT_HERSHEY_COMPLEX,
            offset=6,
        )

    return annotated

def draw_tracked_frame(frame, results, model):

    annotated = frame.copy()

    if results is None or len(results) == 0:
        return annotated

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return annotated

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(classes)
    names = model.names

    for box, cls_id, conf_val, obj_id in zip(xyxy, classes, confidences, ids):
        x1, y1, x2, y2 = map(int, box)
        label = names[int(cls_id)]
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(
            annotated, (x1, y1, w, h), l=30, t=4, rt=1, colorR=color, colorC=[255, 0, 255]
        )

        id_str = f" #{int(obj_id)}" if obj_id is not None else ""
        text = f"{label.replace('_', ' ').title()} {conf_val:.0%}{id_str}"
        
        cvzone.putTextRect(
            annotated,
            text,
            (max(0, x1), max(30, y1 - 10)),
            scale=0.8,
            thickness=1,
            colorT=(255, 255, 255),
            colorR=color,
            font=cv2.FONT_HERSHEY_COMPLEX,
            offset=6,
        )

    return annotated
