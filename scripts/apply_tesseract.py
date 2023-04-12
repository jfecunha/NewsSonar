"""Script that applies Tesseract OCR into news pictures."""
import json
import glob

from pathlib import Path

import pytesseract
import cv2

from joblib import Parallel, delayed
from joblib import parallel_backend

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'


def apply_ocr_on_file(img_path: str) -> None:
    """_summary_

    Parameters
    ----------
    img_path : str
        Image path.
    """
    img = cv2.imread(img_path)

    try:
        # Preprocessing before fed to OCR engine
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(e)
        print(f"Error on image:{img_path}")
        return None

    # Get verbose data including boxes, confidences, line and page numbers
    custom_config = r'--oem 3 --psm 11'
    output = pytesseract.image_to_data(
        gray_img,
        output_type=pytesseract.Output.DICT,
        config=custom_config
    )

    source_dir = Path('data/ocr')
    source_dir.mkdir(parents=True, exist_ok=True)
    source_file = Path(source_dir, f"{Path(img_path).stem}.json")

    with open(source_file, 'w') as file:
        json.dump(output, file)


if __name__ == "__main__":

    image_paths = []
    for dir_ in glob.glob("./data/crop/*"):
        for file in glob.glob(f"{Path(dir_).as_posix()}/*"):
            image_paths.append(file)

    print(len(image_paths))

    with parallel_backend('threading', n_jobs=-1):
        Parallel(n_jobs=-1)(delayed(apply_ocr_on_file)(file) for file in image_paths[0:2])
