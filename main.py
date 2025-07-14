import numpy as np
import cv2
import pytesseract
from PIL import Image
import pandas as pd
import json
import csv
import os

image_path = "Etiquetas Integra (2)/KEEP FROZEN/2260325__integra.jpg"

image_name='2260325__integra.jpg'

# Read the image
image = cv2.imread(image_path)

# print(image)
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

# cv2.imwrite('r_img.jpg',rotated_image)

#resize image
size=2
new_h, new_w = h*size , w*size

resized_image = cv2.resize(rotated_image, (new_w, new_h))

# resized_image = cv2.normalize(rotated_image.astype('float32'), None, 0, 255, cv2.NORM_MINMAX)
# resized_image = resized_image.astype('uint8')

# cv2.imwrite('resized0_image.jpg', resized_image)

gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]

])

sharpened = cv2.filter2D(gray_img, -1, kernel)



clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

contrast_img = clahe.apply(sharpened)

cv2.imwrite('contrast_img.jpg', contrast_img)

ocr_data = pytesseract.image_to_data(sharpened, output_type=pytesseract.Output.DATAFRAME)

filtered_data = ocr_data[(ocr_data.conf != -1) & (ocr_data.text.str.strip() != '')]

high_conf = filtered_data[filtered_data.conf > 50]

text_conf_dict = {
    row['text']: float(row['conf'])
    for _, row in filtered_data.iterrows()
}

dict_str = json.dumps(text_conf_dict, indent=4)

csv_filename = 'output.csv' 

file_exists = os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)

    if not file_exists:
        writer.writerow(['image_name', 'text_with_confidence'])

    writer.writerow([image_name, dict_str])

