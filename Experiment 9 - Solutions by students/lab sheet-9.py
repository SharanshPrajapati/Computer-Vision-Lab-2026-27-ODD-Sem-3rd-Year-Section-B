import os
import time
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------------------
# Step 1 & 3: Acquire Standard Dataset Image (MS COCO Sample)
# -------------------------------------------------------------
IMAGE_PATH = "coco_sample.jpg"
IMAGE_URL = "https://ultralytics.com/images/bus.jpg"  # Standard COCO validation sample

if not os.path.exists(IMAGE_PATH):
    print("Downloading MS COCO benchmark test image...")
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)
    print("Download complete.")

# -------------------------------------------------------------
# Step 2: Load Pre-trained Deep Learning Model (YOLOv8 Nano)
# -------------------------------------------------------------
# Pre-trained on MS COCO (80 object categories)
print("Loading pre-trained YOLOv8n model...")
model = YOLO("yolov8n.pt")

# -------------------------------------------------------------
# Step 4 & 5: Preprocessing & Inference
# -------------------------------------------------------------
start_time = time.time()
# The predict method internally normalizes, resizes (640x640), and runs NMS
results = model.predict(source=IMAGE_PATH, conf=0.35, save=False)
inference_time = (time.time() - start_time) * 1000  # in milliseconds

result = results[0]
boxes = result.boxes
class_names = result.names

# Load image via OpenCV for custom visualization
orig_img = cv2.imread(IMAGE_PATH)
annotated_img = orig_img.copy()

print(f"\n--- Inference Summary ---")
print(f"Total Inference Time: {inference_time:.2f} ms")
print(f"Objects Detected: {len(boxes)}")

# -------------------------------------------------------------
# Step 6 & 7: Parse Bounding Boxes, Class Labels, and Scores
# -------------------------------------------------------------
detected_objects = []

for box in boxes:
    # Coordinates (xmin, ymin, xmax, ymax)
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    confidence = float(box.conf[0])
    class_id = int(box.cls[0])
    label_text = f"{class_names[class_id]}: {confidence:.2f}"

    detected_objects.append(
        {"class": class_names[class_id], "conf": confidence, "box": (x1, y1, x2, y2)}
    )

    # Draw Bounding Box (Step 6)
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Label Background Tag
    (txt_w, txt_h), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )
    cv2.rectangle(
        annotated_img, (x1, y1 - 22), (x1 + txt_w, y1), (0, 255, 0), -1
    )
    cv2.putText(
        annotated_img,
        label_text,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    print(f"Detected: {label_text} at [{x1}, {y1}, {x2}, y2]")

# Overlay Telemetry HUD
cv2.putText(
    annotated_img,
    f"Latency: {inference_time:.1f}ms | Count: {len(boxes)}",
    (15, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2,
)

# -------------------------------------------------------------
# Display Outputs
# -------------------------------------------------------------
# Convert BGR to RGB for Matplotlib visualization (works in Colab/Jupyter/Local)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Input Image")
plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Detected Objects (YOLOv8 + COCO)")
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

# Save rendered detection
cv2.imwrite("detected_output.jpg", annotated_img)
print("Annotated image saved as detected_output.jpg")