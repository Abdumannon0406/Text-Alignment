import cv2
import numpy as np
import pytesseract
import json
import os
import csv

# === Load Image ===
image_path = "Etiquetas Integra (2)/Farm Raised/2248140__integra.jpg"
image = cv2.imread(image_path)

(h, w) = image.shape[:2]

cx, cy = (w // 2, h // 2)

angle = 180

theta = np.radians(angle)

cos = np.cos(theta)
sin = np.sin(theta)

tx= (1 - cos) * cx - sin * cy
ty= sin * cx + (1 - cos) * cy

M = np.array([
    [cos, sin, tx],
    [-sin, cos, ty]
], dtype=np.float32)


rotated_image = cv2.warpAffine(image, M, (w, h))

cv2.imwrite("rotated_img.jpg",rotated_image)


text=pytesseract.image_to_string(rotated_image)

print(text)
