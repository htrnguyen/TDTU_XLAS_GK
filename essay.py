import cv2
import matplotlib.pyplot as plt
import numpy as np

image_input_1 = cv2.imread("./input1.jpg")
image_input_2 = cv2.imread("./input2.png")

# --------------- 1a ---------------
# Convert background to black
image_a = image_input_1.copy()
color_bg = np.array([127, 179, 216])
diff = cv2.absdiff(image_a, color_bg)
mask = cv2.inRange(diff, np.array([0, 0, 0]), np.array([70, 80, 100]))
black_bg = np.zeros_like(image_a)
image_a[mask == 255] = [0, 0, 0]

# List lower and upper
color_dict_HSV = {
    "yellow": [[35, 255, 255], [25, 50, 70]],
    "orange": [[24, 255, 255], [10, 50, 70]],
    "red": [[180, 255, 255], [159, 50, 70]],
    "blue": [[128, 255, 255], [90, 50, 70]],
    "green": [[89, 255, 255], [36, 50, 70]],
    "purple": [[158, 255, 255], [129, 50, 70]],
}
# Convert to HSV
image_hsv = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV)

# Process each color
image_outputs = []
for color, (upper, lower) in color_dict_HSV.items():
    mask = cv2.inRange(image_hsv, np.array(lower), np.array(upper))
    image_output = cv2.bitwise_and(image_a, image_a, mask=mask)
    # Save image
    cv2.imwrite(f"./output_1a_{color}.jpg", image_output)

# --------------- 1b ---------------
image_gray_b = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
image_binary = cv2.threshold(image_gray_b, 0, 255, cv2.THRESH_BINARY_INV)[1]
kernel = np.ones((4, 6), np.uint8)
image_output_b = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel)
# Save image
cv2.imwrite(f"./output_1b.jpg", image_output_b)

# --------------- 2 ---------------
image_2 = cv2.imread("./input2.png")
image_gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

# Get image ROI (region of interest)
h, w = image_gray_2.shape
roi_start = int(h * 0.35)
roi = image_gray_2[roi_start:h, int(w - 0.6 * w) :]  # 35% of height and 60% of width

# Process remove noise
thresh_roi = cv2.threshold(roi, 30, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
opening_roi = cv2.morphologyEx(thresh_roi, cv2.MORPH_OPEN, kernel, iterations=1)
contours, _ = cv2.findContours(opening_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
min_area = 50
contours_to_remove = [cnt for cnt in contours if cv2.contourArea(cnt) < min_area]
for cnt in contours_to_remove:
    cv2.drawContours(opening_roi, [cnt], 0, 0, thickness=cv2.FILLED)
opening_roi = cv2.dilate(opening_roi, kernel, iterations=1)

# Process image gray
thresh_img = cv2.adaptiveThreshold(
    image_gray_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2
)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
opening_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
image_gray = cv2.subtract(thresh_img, opening_img)

# Combine image gray and roi
image_process = image_gray
image_process[roi_start:h, int(w - 0.6 * w) :] = opening_roi

# Find contours and draw rectangle
contours, _ = cv2.findContours(
    image_process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
for cnt in contours:
    if cv2.contourArea(cnt) > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_2, (x, y), (x + w, y + h), (0, 255, 0), 2)
# Save image
cv2.imwrite(f"./output_2.png", image_2)
