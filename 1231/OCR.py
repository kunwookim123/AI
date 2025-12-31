import cv2
from pytesseract import pytesseract as pyt

img = cv2.imread("image/ocr3.png", cv2.IMREAD_GRAYSCALE)

pyt.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ret, binary = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
text = pyt.image_to_string(binary, lang="eng")
print(text)