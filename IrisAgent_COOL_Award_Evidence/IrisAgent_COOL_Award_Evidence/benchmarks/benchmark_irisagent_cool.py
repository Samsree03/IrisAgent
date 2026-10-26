import os
import sys
import json
import time
import platform
import statistics
import numpy as np
import cv2

IMAGE = "/home/ubuntu/irisagent_test_frame.jpg"
RAW_OUTPUT = "/home/ubuntu/irisagent_yolo_raw_output.npy"
WARMUP = 10
REPEATS = 100
CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.4

frame = cv2.imread(IMAGE)
if frame is None:
    raise RuntimeError(f"Cannot read image: {IMAGE}")

raw = np.load(RAW_OUTPUT)
if raw.shape != (1, 7, 8400):
    raise RuntimeError(f"Unexpected YOLO output shape: {raw.shape}")

h, w = frame.shape[:2]
xf, yf = w / 640.0, h / 640.0

def preprocess():
    img = cv2.cvtColor(
        cv2.resize(frame, (640, 640)),
        cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1))[None, ...])

def postprocess():
    out = np.squeeze(raw)
    if out.ndim == 2 and out.shape[0] < out.shape[1]:
        out = out.T

    scores = out[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(out.shape[0]), class_ids]
    mask = confidences >= CONF_THRESHOLD

    if not np.any(mask):
        return []

    filtered_boxes = out[mask, :4]
    filtered_confs = confidences[mask]
    filtered_class_ids = class_ids[mask]

    raw_boxes = []
    for cx, cy, bw, bh in filtered_boxes:
        left = int((cx - bw / 2) * xf)
        top = int((cy - bh / 2) * yf)
        width = int(bw * xf)
        height = int(bh * yf)
        raw_boxes.append([max(0, left), max(0, top), width, height])

    indices = cv2.dnn.NMSBoxes(
        raw_boxes,
        filtered_confs.tolist(),
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    if len(indices) == 0:
        return []

    indices = indices.flatten() if hasattr(indices, "flatten") else indices
    return [
        {
            "box": raw_boxes[int(i)],
            "class_id": int(filtered_class_ids[int(i)]),
            "confidence": round(float(filtered_confs[int(i)]), 7)
        }
        for i in indices
    ]

def measure(fn):
    for _ in range(WARMUP):
        fn()

    samples = []
    result = None
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        result = fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)

    return {
        "mean_ms": round(statistics.mean(samples), 4),
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(float(np.percentile(samples, 95)), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
        "throughput_ops_s": round(1000 / statistics.mean(samples), 2),
    }, result

pre_stats, blob = measure(preprocess)
post_stats, detections = measure(postprocess)

result = {
    "architecture": platform.machine(),
    "opencv_version": cv2.__version__,
    "opencv_module": cv2.__file__,
    "image_shape": list(frame.shape),
    "raw_yolo_output_shape": list(raw.shape),
    "warmup_iterations": WARMUP,
    "measured_iterations": REPEATS,
    "confidence_threshold": CONF_THRESHOLD,
    "nms_threshold": NMS_THRESHOLD,
    "preprocessing": pre_stats,
    "postprocessing_including_nms": post_stats,
    "detection_count": len(detections),
    "detections": detections,
    "preprocessing_blob_shape": list(blob.shape),
    "preprocessing_blob_dtype": str(blob.dtype),
}

output_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/benchmark_result.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f"\nSaved benchmark results: {output_path}")
