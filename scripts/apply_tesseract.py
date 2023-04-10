"""Script that applies Tesseract OCR into news pictures."""
from PIL import Image
import os

import pytesseract

# Get verbose data including boxes, confidences, line and page numbers
print(pytesseract.image_to_data(Image.open('cmjornal-20200901070349.png')))
