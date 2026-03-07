import numpy as np

# from data.yaml to dict
EXPECTED_EQUIPMENT = {
    "hospital_bed": {"priority": "Critical", 
                     "score": 3, 
                     "description": "Hospital Bed"
                     },
                     
    "monitor": {"priority": "Critical", 
                "score": 3, 
                "description": "Monitor"
                },

    "infusion_pole": {
                "priority": "Important",
                "score": 2,
                "description": "Infusion Pole / IV Stand",
               },

    "stretcher": {"priority": "Supplementary", 
                  "score": 1, 
                  "description": "Stretcher"
                  },

    "wheelchair": {
                "priority": "Supplementary",
                "score": 1,
                "description": "Wheelchair",
               }
}

# looping through dict to calculate max score
MAX_ROOM_SCORE = sum(eq["score"] for eq in EXPECTED_EQUIPMENT.values())

def summarize(results, model):

    if results is None or not isinstance(results, list) or len(results) == 0:
        return empty_summary()

    class_names = model.names

    total = 0
    class_counts = {}
    class_conf_lists = {}
    all_confidences = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes

        # GPU-safe tensor conversion
        classes = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        total += len(classes)
        all_confidences.extend(confidences)

        for cls_id, conf in zip(classes, confidences):
            label = class_names[int(cls_id)]
            class_counts[label] = class_counts.get(label, 0) + 1
            class_conf_lists.setdefault(label, []).append(float(conf))

    if total == 0:
        return empty_summary()

    confidences_np = np.array(all_confidences)
    avg_conf = float(confidences_np.mean())
    max_conf = float(confidences_np.max())
    min_conf = float(confidences_np.min())

    # class confidence breakdown
    class_details = {}
    for label in class_counts:
        confs = np.array(class_conf_lists[label])
        class_details[label] = {
            "count": class_counts[label],
            "avg_confidence": round(float(confs.mean()), 3),
            "min_confidence": round(float(confs.min()), 3),
            "max_confidence": round(float(confs.max()), 3),
        }

    # Equipment checklist
    equipment_status = _build_equipment_status(class_counts)

    # Room health assessment
    room_assessment = evaluate_room_health(class_counts)

    # Low-confidence alerts
    low_conf_threshold = 0.4
    low_conf_alerts = [
        {"class": label, "confidence": round(c, 3)}
        for label, confs in class_conf_lists.items()
        for c in confs
        if c < low_conf_threshold
    ]

    return {
        "total_detections": total,
        "unique_classes": len(class_counts),
        "class_counts": class_counts,
        "avg_confidence": round(avg_conf, 3),
        "max_confidence": round(max_conf, 3),
        "min_confidence": round(min_conf, 3),
        "class_details": class_details,
        "equipment_status": equipment_status,
        "room_assessment": room_assessment,
        "low_confidence_alerts": low_conf_alerts,
    }

def evaluate_room_health(class_counts):
    score = 0
    missing_critical = []
    missing_important = []
    present = []

    for eq_name, eq_info in EXPECTED_EQUIPMENT.items():
        if eq_name in class_counts:
            score += eq_info["score"]
            present.append(eq_info["description"])
        else:
            if eq_info["priority"] == "Critical":
                missing_critical.append(eq_info["description"])
            elif eq_info["priority"] == "Important":
                missing_important.append(eq_info["description"])

    pct = (score / MAX_ROOM_SCORE) * 100 if MAX_ROOM_SCORE > 0 else 0

    if pct >= 80:
        status = "Well Equipped"
        level = "success"
    elif pct >= 50:
        status = "Moderately Equipped"
        level = "warning"
    else:
        status = "Under Equipped"
        level = "error"

    return {
        "score": score,
        "max_score": MAX_ROOM_SCORE,
        "percentage": round(pct, 1),
        "status": status,
        "level": level,
        "present": present,
        "missing_critical": missing_critical,
        "missing_important": missing_important,
    }

def _build_equipment_status(class_counts):
    status = {}
    for eq_name, eq_info in EXPECTED_EQUIPMENT.items():
        status[eq_name] = {
            "detected": eq_name in class_counts,
            "count": class_counts.get(eq_name, 0),
            "priority": eq_info["priority"],
            "description": eq_info["description"],
        }
    return status

def empty_summary():
    return {
        "total_detections": 0,
        "unique_classes": 0,
        "class_counts": {},
        "avg_confidence": 0,
        "max_confidence": 0,
        "min_confidence": 0,
        "class_details": {},
        "equipment_status": _build_equipment_status({}),
        "room_assessment": evaluate_room_health({}),
        "low_confidence_alerts": [],
    }
