import cv2 as cv
import pytesseract
import numpy as np

# Load image
image = cv.imread("Etiquetas Integra (2)/KEEP FROZEN/2260330__integra.jpg")

# Step 1: Rotate 180 degrees to make text upright
rotated = cv.rotate(image, cv.ROTATE_180)

# Step 2: Convert to grayscale
gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)

# Step 3: Improve contrast using CLAHE
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
contrast = clahe.apply(gray)

# Step 4: Apply thresholding
_, thresh = cv.threshold(contrast, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Step 5: Sharpen the image
blurred = cv.GaussianBlur(thresh, (0, 0), 3)
sharpened = cv.addWeighted(thresh, 1, blurred, -0.1, 0)

# Optional: show intermediate steps
cv.imshow("Sharpened", sharpened)
cv.waitKey(0)
cv.destroyAllWindows()

# Step 6: OCR with Tesseract
custom_config = r'--oem 1 --psm 6'
text = pytesseract.image_to_string(sharpened, config=custom_config)

print("OCR Result:\n", text)
