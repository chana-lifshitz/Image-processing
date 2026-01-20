import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


#Q1
def create_gradient_image(height, width):
    img = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            img[y, x] = int(((x + y) / (width + height - 2)) * 255)

    return img
img = create_gradient_image(255, 255)

plt.imshow(img, cmap="gray")
plt.title("Gradient Image")
plt.axis("off")
plt.show()

#Q2,Q3
def brighten(img, b, func):
    if func == "np":
        return np.add(img, b)
    elif func == "cv2":
        return cv2.add(img, b)
    else:
        raise ValueError("func must be 'np' or 'cv2'")

import matplotlib.pyplot as plt

img = create_gradient_image(255, 255)

bright_np = brighten(img, 60, "np")
bright_cv2 = brighten(img, 60, "cv2")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Brighten using np.add")
plt.imshow(bright_np, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Brighten using cv2.add")
plt.imshow(bright_cv2, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

#Q4
def create_low_contrast_image(fg, bg, height=256, width=256):
    img = np.full((height, width), fg, dtype=np.uint8)

    center = (width // 2, height // 2)
    radius = min(height, width) // 4

    cv2.circle(img, center, radius, bg, thickness=-1)

    return img

img_low_contrast = create_low_contrast_image(fg=100, bg=105)

plt.imshow(img_low_contrast, cmap="gray", vmin=0, vmax=255)
plt.title("Low Contrast Image (fg=100, bg=105)")
plt.colorbar(label="Pixel Value")  
plt.axis("off")
plt.show()

#Q5
def create_gradient_image(height, width):
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x] = int(((x + y) / (width + height - 2)) * 255)
    return img


def brighten(img, b, func):
    if func == "np":
        return np.add(img, b)
    elif func == "cv2":
        return cv2.add(img, b)
    else:
        raise ValueError("func must be 'np' or 'cv2'")


img = create_gradient_image(255, 255)
bright_img = brighten(img, 60, "cv2")

min_val, max_val, _, _ = cv2.minMaxLoc(bright_img)
mean_val = np.mean(bright_img)

stretch_factor = 255 / (max_val - min_val)

print("Before normalization:")
print(f"Min pixel value: {min_val}")
print(f"Max pixel value: {max_val}")
print(f"Mean pixel value: {mean_val:.2f}")
print(f"Max stretch factor (255 / (max-min)): {stretch_factor:.2f}")
def normalize(src_image):
    min_val, max_val, _, _ = cv2.minMaxLoc(src_image)
    mean_val = np.mean(src_image)

    print("\nInside normalize():")
    print(f"Min: {min_val}, Max: {max_val}, Mean: {mean_val:.2f}")

    src_float = src_image.astype(np.float32)
    scale = 255.0 / (max_val - min_val)
    dst_float = (src_float - min_val) * scale
    dst = np.clip(dst_float, 0, 255).astype(np.uint8)

    return dst
normalized_img = normalize(bright_img)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Before Normalization")
plt.imshow(bright_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("After Normalization")
plt.imshow(normalized_img, cmap="gray")
plt.axis("off")
plt.show()

#Q6
modified_img = bright_img.copy()

modified_img[0, 0] = 0         
modified_img[1, 1] = 255     

normalized_modified = normalize(modified_img)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Before normalization")
plt.imshow(bright_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("After pixel modification")
plt.imshow(modified_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("After normalization")
plt.imshow(normalized_modified, cmap="gray")
plt.axis("off")

plt.show()

#Q7
img_color = cv2.imread("image.jpg")

if len(img_color.shape) == 3:
    img_gray = (
        0.299 * img_color[:, :, 2] +
        0.587 * img_color[:, :, 1] +
        0.114 * img_color[:, :, 0]
    ).astype(np.uint8)
else:
    img_gray = img_color

hist = [0] * 256

height, width = img_gray.shape
for y in range(height):
    for x in range(width):
        pixel_value = img_gray[y, x]
        hist[pixel_value] += 1

plt.figure(figsize=(8, 4))
plt.bar(range(256), hist)
plt.title("Grayscale Histogram")
plt.xlabel("Gray level (0–255)")
plt.ylabel("Number of pixels")
plt.show()
