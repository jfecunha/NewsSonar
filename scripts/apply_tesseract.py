"""Script that applies Tesseract OCR into news pictures."""
import json

from PIL import Image
from pathlib import Path

import pytesseract

img = Image.open('cmjornal-20200901070349.png').convert("RGB")

# Get verbose data including boxes, confidences, line and page numbers
output = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

source_file = Path('data/')
source_file.mkdir(parents=True, exist_ok=True)

with open(Path(source_file, 'result_dummy.json'), 'w') as file:
    json.dump(output, file)
