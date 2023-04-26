"""Script that applies Tesseract OCR into news pictures."""
import json
import glob

from pathlib import Path
from PIL import Image, ImageEnhance

import pytesseract
import cv2
import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from joblib import parallel_backend

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'


def process_image(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # inverse binary image, to ensure text region is in white
    # because contours are found for objects in white
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel_length = 10
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    dilate = cv2.dilate(th, horizontal_kernel, iterations=1)
    return dilate

def run_ocr(img_path):
    raw_img = cv2.imread(img_path)
    processed_img = process_image(raw_img)
    contours = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    custom_config = r'--oem 3 --psm 7'
    file_ocr = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        # filter noise
        if area <= 15000 and area > 200:

            crop_margin = 3
            croping_shape = (x-crop_margin, y-crop_margin, (x+w) + crop_margin, (y+h) + crop_margin)
            raw_img_pil = Image.fromarray(np.uint8(raw_img))
            croped_img = raw_img_pil.crop(croping_shape)

            # Increase contrast and convert to grayscale
            croped_img_proc = ImageEnhance.Contrast(croped_img)
            croped_img_proc = croped_img_proc.enhance(1.5).convert('L')

            ocr_result = pytesseract.image_to_data(
                croped_img_proc,
                output_type=pytesseract.Output.DICT,
                config=custom_config)

            temp_df = dict()
            temp_df['text'] = [' '.join(ocr_result["text"])]
            temp_df['left'] = [x]
            temp_df['top'] = [y]
            temp_df['right'] = [x+w]
            temp_df['bottom'] = [y+h]
            temp_df['width'] = [w]
            temp_df['height'] = [h]

            subwords = []
            width = w / len(''.join(temp_df['text']))
            for _, word in enumerate(ocr_result["text"]):
                word_meta = dict()
                word_meta['word'] = word
                word_meta['left'] = x
                word_meta['top'] = y
                word_meta['width'] = width
                word_meta['height'] = h
                word_meta['right'] = [x+width]
                word_meta['bottom'] = [y+h]
                x = x + width
                subwords.append(word_meta)

            temp_df['words'] = subwords

            file_ocr.append(temp_df)
    return file_ocr


def apply_ocr_on_file(img_path: str) -> None:
    """Apply and save OCR result.

    Parameters
    ----------
    img_path : str
        Image path.
    """
    try:

        output = run_ocr(img_path)
    except Exception as e:
        print(e)
        print(f"Error on image:{img_path}")
        return None

    source_dir = Path('data', 'ocr')
    source_dir.mkdir(parents=True, exist_ok=True)
    source_file = Path(source_dir, f"{Path(img_path).stem}.csv")
    output.to_csv(source_file, index=False)


if __name__ == "__main__":

    image_paths = []
    for dir_ in glob.glob("./data/crop/*"):
        for file in glob.glob(f"{Path(dir_).as_posix()}/*"):
            image_paths.append(file)

    print(len(image_paths))

    with parallel_backend('threading', n_jobs=-1):
        Parallel(n_jobs=-1)(delayed(apply_ocr_on_file)(file) for file in image_paths)
