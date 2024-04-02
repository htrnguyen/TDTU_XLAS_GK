import cv2
import numpy as np

image_1 = cv2.imread("./input1.jpg")


# ------------------- 1 -------------------
def get_common_color_RGB(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate the histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    # Find the color with the most pixels
    most_common_color_index = np.unravel_index(np.argmax(hist), hist.shape)
    # Convert the index to actual color value
    most_common_color = [(index / 8) * 255 for index in most_common_color_index]
    # Convert the color from HSV to BGR
    most_common_color_bgr = cv2.cvtColor(
        np.uint8([[most_common_color]]), cv2.COLOR_HSV2BGR
    )[0][0]
    # Convert the color from BGR to RGB
    most_common_color_rgb = most_common_color_bgr[::-1]

    return most_common_color_rgb


# ------------------- 1a -------------------
image_1a = image_1.copy()
# Convert background color to black
background_color = get_common_color_RGB(image_1a)
diff = cv2.absdiff(image_1a, background_color)
mask = cv2.inRange(diff, np.array([0, 0, 0]), np.array([130, 85, 110]))
black_bg = np.zeros_like(image_1a)
image_1a[mask == 255] = [0, 0, 0]

# List lower and upper
color_dict_HSV = {
    "yellow": [[35, 255, 255], [25, 50, 70]],
    "orange": [[12, 255, 255], [10, 100, 100]],
    "red": [[180, 255, 255], [159, 50, 70]],
    "blue": [[128, 255, 255], [90, 50, 70]],
    "green": [[89, 255, 255], [36, 50, 70]],
    "purple": [[158, 255, 255], [129, 50, 70]],
}
# Convert to HSV
image_hsv = cv2.cvtColor(image_1a, cv2.COLOR_BGR2HSV)

# Process each color
for color, (upper, lower) in color_dict_HSV.items():
    mask = cv2.inRange(image_hsv, np.array(lower), np.array(upper))
    image_output = cv2.bitwise_and(image_1a, image_1a, mask=mask)

    # Save the image
    cv2.imwrite(f"./output_1a_{color}.jpg", image_output)

# ------------------- 1b -------------------
image_gray_1b = cv2.cvtColor(image_1a, cv2.COLOR_BGR2GRAY)
thresh_binary = cv2.threshold(image_gray_1b, 0, 255, cv2.THRESH_BINARY_INV)[1]
kernel = np.ones((6, 6), np.uint8)
image_1b = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, kernel)
cv2.imwrite("./output_1b.jpg", image_1b)


# ------------------- 2 -------------------
def find_ROI(image):
    _, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    def contour_area(contour):
        _, _, w, h = cv2.boundingRect(contour)
        return w * h

    largest_contour = max(contours, key=contour_area)
    x, y, w, h = cv2.boundingRect(largest_contour)

    noisy_region = image[y : y + h, x : x + w]

    return noisy_region, (x, y, w, h)


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive threshold
    thresh_binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 2
    )
    kernel_lines = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 1))
    # Create an image with only the lines
    img_lines = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, kernel_lines)
    # Subtract the lines from the original image
    process_img = cv2.subtract(thresh_binary, img_lines)

    return process_img


def remove_small_contours(process_img):
    # Find contours
    contours, _ = cv2.findContours(process_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Remove small contours have an area less than 100
        if cv2.contourArea(contour) < 100:
            cv2.drawContours(process_img, [contour], 0, 0, cv2.FILLED)

    return process_img


def erode_and_open(process_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    process_img = cv2.erode(process_img, kernel, iterations=2)
    process_img = cv2.morphologyEx(process_img, cv2.MORPH_OPEN, kernel, iterations=1)

    return process_img


def insert_roi(process_img, roi, x, y, h, w):
    # Preprocess the ROI image and insert it into the processed image
    roi = cv2.bitwise_not(roi)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)

    process_img[y : y + h, x : x + w] = cv2.resize(roi, (w, h))

    return process_img


def draw_bounding_boxes(image, process_img):
    # Find contours and draw bounding boxes
    contours = cv2.findContours(
        process_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    for cnt in contours:
        if cv2.contourArea(cnt) > 150:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y - 2), (x + w, y + h), (0, 255, 0), 2)

    return image


image_path = "./input2.png"
process_img = preprocess_image(image_path)
process_img = remove_small_contours(process_img)
process_img = erode_and_open(process_img)

roi, (x, y, h, w) = find_ROI(~process_img)
process_img = insert_roi(process_img, roi, x, y, h, w)

image = cv2.imread(image_path)
image = draw_bounding_boxes(image, process_img)

cv2.imwrite("./output_2.png", image)
