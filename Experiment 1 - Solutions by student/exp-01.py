import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# =========================================================================
# Step 1: Path Setup and Directory Creation
# =========================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(script_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

image_path = os.path.join(script_dir, "colourful.jpeg")

# =========================================================================
# Step 2: Load Color Image and Display (OpenCV + Matplotlib)
# =========================================================================
image = cv2.imread(image_path)

if image is None:
    print(f"\nError: Could not load image from:\n{image_path}")
    print("Ensure 'colourful.jpeg' (or .jpg) is in the same directory.")
    sys.exit()

print("Image loaded successfully.")

# Display via OpenCV window
cv2.imshow("Step 2 - OpenCV Native Window (Press Any Key)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert BGR to RGB for Matplotlib rendering
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# =========================================================================
# Step 3: Examine Image Properties
# =========================================================================
height, width = image.shape[:2]
channels = image.shape[2] if len(image.shape) == 3 else 1

print("\n--- STEP 3: IMAGE PROPERTIES ---")
print("Dimensions  :", image.shape)
print("Resolution  :", f"{width} x {height}")
print("Channels    :", channels)
print("Data Type   :", image.dtype)
print("Total Pixels:", height * width)

# =========================================================================
# Step 4: Save Different Formats and Compare Output Sizes
# =========================================================================
out_jpg = os.path.join(outputs_dir, "output.jpg")
out_png = os.path.join(outputs_dir, "output.png")

cv2.imwrite(out_jpg, image)
cv2.imwrite(out_png, image)

print("\n--- STEP 4: FILE SIZES ---")
print("JPEG (Lossy)   :", os.path.getsize(out_jpg), "bytes")
print("PNG  (Lossless):", os.path.getsize(out_png), "bytes")

# =========================================================================
# Step 5: Color Space Conversions (Grayscale, HSV, LAB)
# =========================================================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# =========================================================================
# Step 6: Geometric Transformations (Resize, Rotate, Flips)
# =========================================================================
resized = cv2.resize(image, None, fx=0.5, fy=0.5)

center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

horizontal_flip = cv2.flip(image, 1)
vertical_flip = cv2.flip(image, 0)

# =========================================================================
# Step 7: Image Complement (Negative)
# =========================================================================
negative = 255 - image

# =========================================================================
# Step 8: Crop Region of Interest (ROI) and Statistical Analysis
# =========================================================================
x1, y1 = int(width * 0.1), int(height * 0.1)
x2, y2 = int(width * 0.6), int(height * 0.6)
roi = image[y1:y2, x1:x2]

print("\n--- STEP 8: ROI PROPERTIES ---")
if roi.size > 0:
    print("ROI Dimensions:", roi.shape)
    print("ROI Mean      :", round(float(np.mean(roi)), 2))
    print("ROI Minimum   :", np.min(roi))
    print("ROI Maximum   :", np.max(roi))
else:
    print("ROI is empty.")

# =========================================================================
# Step 9: Display All Operations via Matplotlib Grid
# =========================================================================
plt.figure(figsize=(16, 10))

plots = [
    ("Original RGB", rgb, None),
    ("Grayscale", gray, "gray"),
    ("HSV Color Space", hsv, None),
    ("LAB Color Space", lab, None),
    ("Negative (Inverted)", cv2.cvtColor(negative, cv2.COLOR_BGR2RGB), None),
    ("Resized (50%)", cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), None),
    ("Rotated (45°)", cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), None),
    ("Horizontal Flip", cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB), None),
    ("Vertical Flip", cv2.cvtColor(vertical_flip, cv2.COLOR_BGR2RGB), None),
    ("Cropped ROI", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), None)
]

for index, (title, img_data, color_map) in enumerate(plots, 1):
    plt.subplot(2, 5, index)
    if color_map:
        plt.imshow(img_data, cmap=color_map)
    else:
        plt.imshow(img_data)
    plt.title(title, fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()

# =========================================================================
# Step 10: Observations & Preprocessing Significance
# =========================================================================
print("\n--- STEP 10: OBSERVATIONS & PREPROCESSING SIGNIFICANCE ---")
print("1. Color Spaces: Grayscale reduces 3-channel overhead; HSV/LAB decouple lighting from color.")
print("2. Formats     : PNG preserves exact pixel arrays; JPEG offers smaller file sizes.")
print("3. Geometry    : Resizing speeds up pipeline inference; Flipping provides data augmentation.")
print("4. ROI Focus   : Cropping eliminates background noise, directing compute to relevant targets.")