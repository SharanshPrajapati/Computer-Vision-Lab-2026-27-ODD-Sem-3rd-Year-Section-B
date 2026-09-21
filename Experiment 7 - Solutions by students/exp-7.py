import os
import urllib.request
import cv2
import numpy as np

# -------------------------------------------------------------
# 0. Download 'car-detection.mp4' if not present
# -------------------------------------------------------------
VIDEO_PATH = "car-detection.mp4"
VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4"

if not os.path.exists(VIDEO_PATH):
    print(f"Downloading {VIDEO_PATH}...")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
    print("Download complete.")

# -------------------------------------------------------------
# 1. Initialize Video and Tracking Parameters
# -------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

# Parameters for Lucas-Kanade
feature_params = dict(maxCorners=150, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

ret, first_frame = cap.read()
if not ret:
    print("Error reading video stream.")
    cap.release()
    exit()

# Resize dimensions for clean side-by-side display
disp_w, disp_h = 480, 270

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
trajectory_mask = np.zeros_like(first_frame)

# HSV template for Farnebäck dense flow
hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255

print("Running side-by-side visualization. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # Panel 1: Lucas-Kanade Sparse Flow
    # ---------------------------------------------------------
    lk_vis = frame.copy()

    # Re-sample feature points if tracking drops
    if p0 is None or len(p0) < 20:
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                trajectory_mask = cv2.line(
                    trajectory_mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2
                )
                lk_vis = cv2.circle(lk_vis, (int(a), int(b)), 4, (0, 0, 255), -1)

            lk_vis = cv2.add(lk_vis, trajectory_mask)
            p0 = good_new.reshape(-1, 1, 2)

    # ---------------------------------------------------------
    # Panel 2: Farnebäck Dense Flow
    # ---------------------------------------------------------
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    dense_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Update previous frame
    prev_gray = frame_gray.copy()

    # ---------------------------------------------------------
    # Resize and Combine Panels
    # ---------------------------------------------------------
    raw_resized = cv2.resize(frame, (disp_w, disp_h))
    lk_resized = cv2.resize(lk_vis, (disp_w, disp_h))
    dense_resized = cv2.resize(dense_vis, (disp_w, disp_h))

    # Add text labels
    cv2.putText(raw_resized, "1. Original Feed", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(lk_resized, "2. Lucas-Kanade (Sparse)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(dense_resized, "3. Farneback (Dense HSV)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Stack all 3 horizontally
    combined_display = np.hstack([raw_resized, lk_resized, dense_resized])

    cv2.imshow("Optical Flow Comparison - car-detection.mp4", combined_display)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
