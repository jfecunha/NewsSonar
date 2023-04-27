"""Module to build dataset."""
import json
import glob
import re
import io
import sys
import logging

from pathlib import Path

parent = Path().absolute().as_posix()
sys.path.insert(0, parent)

from typing import Dict, List
from copy import deepcopy
from PIL import Image, ImageEnhance

import pandas as pd
import numpy as np
import cv2
import pytesseract

from joblib import Parallel, delayed
from joblib import parallel_backend
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from utils import DataExtractor

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'


class DatasetBuilder:
    def __init__(self, overlap_ratio=0.3, train=True) -> None:
        self.overlap_ratio = overlap_ratio
        self.annotation_max_range = 100
        self.train = train
        self.label2id = {
            'None': 0,
            'Title': 1,
            'SubTitle': 2,
            'Category': 3
        }

    def __call__(self, annotation: Dict) -> None:
        if annotation["annotations"][0]["result"]:
            try:
                annotation_scaled = self._process_annotation(annotation)
                file_stem = re.findall(r"-(.*)", Path(annotation["file_upload"]).stem)[0]
                newspaper = file_stem.split("-")[0]
                img_file_path = Path("data", "crop", newspaper, f"{file_stem}.png")
                img = cv2.imread(img_file_path.as_posix())
                file_ocr_data = self.run_ocr(img)
                annotation_processed = self._get_text_from_ocr_df(annotation_scaled, file_ocr_data)
                self._save_processed_annotation(file_stem, annotation_processed)
            except Exception as e:
                print(e)
                return None


    @staticmethod
    def _save_processed_annotation(file_name, annot_data):
        if not isinstance(annot_data, type(None)):
            source_dir = Path("data", "annotations")
            source_dir.mkdir(parents=True, exist_ok=True)
            source_file = Path(source_dir, f"{file_name}.csv")
            annot_data.to_csv(source_file, index=False)

    def _process_annotation(self, annotations):
        """Process annotation to match with OCR coordinates."""
        annotations_scaled = [
            {
                "left": (val["value"]["x"] * val["original_width"])
                / self.annotation_max_range,
                "top": (val["value"]["y"] * val["original_height"])
                / self.annotation_max_range,
                "right": (val["value"]["x"] * val["original_width"])
                / self.annotation_max_range
                + (val["value"]["width"] * val["original_width"])
                / self.annotation_max_range,
                "bottom": (val["value"]["y"] * val["original_height"])
                / self.annotation_max_range
                + (val["value"]["height"] * val["original_height"])
                / self.annotation_max_range,
                "label": val["value"]["rectanglelabels"][0],
            }
            for val in annotations["annotations"][0]["result"]
        ]

        return annotations_scaled

    def _get_text_from_ocr_df(self, annotation: Dict, ocr: Dict) -> List[Dict]:
        """Add text to annotation."""
        if self.train:
            ocr_df = pd.DataFrame(DataExtractor.flat_list([val["words"] for val in ocr]))
        else:
            ocr_df = pd.DataFrame()

        # Remove broken words like None
        ocr_df = ocr_df.dropna(ocr)

        words_metadata = []
        for bb in deepcopy(annotation):
            # Find all ocr_dfs overlapping with ground truth boxes.
            ocr_df["_l"] = ocr_df["left"].apply(lambda x: max(x, bb["left"]))
            ocr_df["_t"] = ocr_df["top"].apply(lambda x: max(x, bb["top"]))
            ocr_df["_r"] = ocr_df["right"].apply(lambda x: min(x, bb["right"]))
            ocr_df["_b"] = ocr_df["bottom"].apply(lambda x: min(x, bb["bottom"]))

            overlapping_words = deepcopy(ocr_df).query("_l < _r and _t < _b")

            if len(overlapping_words) >= 1:
                overlapping_words["int_area"] = overlapping_words.apply(
                    lambda col: (col["_r"] - col["_l"]) * (col["_b"] - col["_t"]),
                    axis=1,
                )

                overlapping_words["word_area"] = overlapping_words.apply(
                    lambda col: col["width"] * col["height"], axis=1
                )

                overlapping_words = overlapping_words.query(
                    "int_area >= @self.overlap_ratio * word_area"
                )

                for _, region in overlapping_words.iterrows():
                    # if OCR has text for the box
                    if region["text"]:
                        words_metadata.append(
                            {
                                "word": region["text"],
                                "label": self.label2id.get(bb["label"], 0),
                                "top": region["top"],
                                "left": region["left"],
                                "right": region["right"],
                                "bottom": region["bottom"],
                                "overlap": True,
                            }
                        )
        words_overlapping = pd.DataFrame(words_metadata)

        if not words_overlapping.empty:
            all_words_df = pd.merge(
                ocr_df,
                words_overlapping,
                how="left",
                right_on=["word", "left", "top", "right", "bottom"],
                left_on=["text", "left", "top", "right", "bottom"],
            )
            all_words_df = (
                all_words_df.copy()
                .query("text != ''")
                .sort_values(by=["top", "right"], ascending=True)
            )
            all_words_df.loc[all_words_df.label.isna(), "label"] = 0

            # NOTE: Must follow (x0, y0, x1, y1)
            all_words_df["bboxes"] = all_words_df.apply(
                lambda row: [row["left"], row["top"], row["right"], row["bottom"]],
                axis=1,
            )
            all_words_df = all_words_df[["text", "bboxes", "label"]]
            all_words_df = all_words_df.dropna()
        else:
            # NOTE: This can be None to don't be saved the output
            all_words_df = None

        return all_words_df

    @staticmethod
    def normalize_bbox(bbox, width, height):
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]

    @staticmethod
    def process_image(img):
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        # inverse binary image, to ensure text region is in white
        # because contours are found for objects in white
        th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel_length = 10
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        dilate = cv2.dilate(th, horizontal_kernel, iterations=1)
        return dilate

    def run_ocr(self, raw_img):
        processed_img = self.process_image(raw_img)
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

                temp = dict()
                temp['text'] = ' '.join(ocr_result["text"])
                temp['left'] = x
                temp['top'] = y
                temp['right'] = x+w
                temp['bottom'] = y+h
                temp['width'] = w
                temp['height'] = h

                if self.train:
                    subwords = []
                    for _, ocr_meta in enumerate(zip(ocr_result["left"], ocr_result["width"], ocr_result["text"], ocr_result["conf"])):
                        # Discard confidences like -1
                        if ocr_meta[3] != -1:
                            word_meta = dict()
                            word_meta['text'] = ocr_meta[2]
                            word_meta['left'] = x + ocr_meta[0]
                            word_meta['top'] = y
                            word_meta['width'] = ocr_meta[1]
                            word_meta['height'] = h
                            word_meta['right'] = (x + ocr_meta[0]) + ocr_meta[1]
                            word_meta['bottom'] = y+h
                            subwords.append(word_meta)
                    temp['words'] = subwords
                file_ocr.append(temp)
        return file_ocr


if __name__ == "__main__":

    logger = logging.getLogger("Data-Prep")
    logging.basicConfig(level=logging.INFO)

    # Load annotations data
    file = open("./annotations/project-2-at-2023-04-06-16-59-565329e3.json")
    annotations = json.load(file)

    logger.info('Processing annotations')
    db = DatasetBuilder()
    with parallel_backend("threading", n_jobs=-1):
        Parallel(n_jobs=-1)(delayed(db)(file) for file in annotations[0:])

    logger.info('Processing dataset')
    # Generate a dataframe with all the data
    aggregated_annotations = pd.DataFrame()
    for index, file in enumerate(glob.glob("./data/annotations/*")):
        # NOTE: This avoids tha string "NA" be considered as NaN
        file_df = pd.read_csv(file, na_filter=False)
        if not file_df.empty:
            source = Path(file).stem.split("-")[0]

            image_path = Path(
                    "data", "crop", source, Path(file).name.replace(".csv", ".png")
                ).as_posix()

            # Convert image to bytes to be stores in pandas dataframe
            image = Image.open(image_path).convert("RGB")
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_data = img_bytes.getvalue()
            temp = pd.DataFrame()
            temp["image_id"] = [index]
            temp["words"] = [list(file_df["text"].values)]
            # Normalize to match LayoutLMv2 range (0, 1000)
            temp["bboxes"] = [[db.normalize_bbox(eval(val), 2000, 2000) for val in file_df["bboxes"]]]
            temp["labels"] = [list(file_df["label"].values)]
            temp["image"] = [img_data]
            temp["image_path"] = [image_path]
            temp["source"] = [source]
            aggregated_annotations = pd.concat([aggregated_annotations, temp], axis=0)

    logger.info('Create partitions')
    # create partitions for model training
    train_df, test_df, _, __ = train_test_split(
        aggregated_annotations,
        aggregated_annotations.source,
        test_size=0.20,
        random_state=42,
        stratify=aggregated_annotations.source,
    )

    logger.info('Saving dataset locally')
    save_dir = Path("data", "processed")
    save_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(Path("data", "processed", "train_data.parquet"), engine='pyarrow')
    test_df.to_parquet(Path("data", "processed", "test_data.parquet"), engine='pyarrow')

    logger.info('Saving dataset to huggingface hub')
    # TODO: try to load directly from pandas
    dataset = load_dataset(
        "parquet",
        data_files={'train': 'train_data.parquet', 'test': 'test_data.parquet'},
        data_dir='./data/processed'
    )

    dataset.push_to_hub("jfecunha/arquivo_news", token='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')
