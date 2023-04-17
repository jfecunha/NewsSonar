"""Module to build dataset."""
import json
import glob
import re
import io

from pathlib import Path
from typing import Dict, List
from copy import deepcopy
from PIL import Image

import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from joblib import parallel_backend
from sklearn.model_selection import train_test_split


class AnnotationsProcessor:
    def __init__(self, overlap_ratio=0.3) -> None:
        self.overlap_ratio = overlap_ratio
        self.annotation_max_range = 100

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
            except IndexError:
                return None
            ocr_data, file_name = self._load_ocr_data(annotation["file_upload"])
            annotation_processed = self._get_text_from_ocr_df(
                annotation_scaled, ocr_data
            )
            self._save_processed_annotation(file_name, annotation_processed)

    @staticmethod
    def _load_ocr_data(annotation_file_path):
        file_stem = re.findall(r"-(.*)", Path(annotation_file_path).stem)[0]
        ocr_file_path = Path("data", "ocr", f"{file_stem}.json")
        file = open(ocr_file_path)
        ocr_data = json.load(file)
        return ocr_data, file_stem

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
        ocr_df = pd.DataFrame(ocr)
        ocr_df["right"] = ocr_df["left"] + ocr_df["width"]
        ocr_df["bottom"] = ocr_df["top"] + ocr_df["height"]

        # Remove broken words like None
        ocr_df = ocr_df.dropna()

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


if __name__ == "__main__":
    # Load annotations data
    file = open("./annotations/project-2-at-2023-04-06-16-59-565329e3.json")
    annotations = json.load(file)

    ap = AnnotationsProcessor()
    with parallel_backend("threading", n_jobs=-1):
        Parallel(n_jobs=-1)(delayed(ap)(file) for file in annotations[0:])

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

            temp_df = pd.DataFrame()
            temp_df["id"] = [index]
            temp_df["words"] = [list(file_df["text"].values)]
            # Normalize to match LayoutLMv2 range (0, 1000)
            temp_df["bboxes"] = [[ap.normalize_bbox(eval(val), 2000, 2000) for val in file_df["bboxes"]]]
            temp_df["labels"] = [list(file_df["label"].values)]
            temp_df["image"] = [img_data]
            temp_df["image_path"] = [image_path]
            temp_df["source"] = [source]
            aggregated_annotations = pd.concat([aggregated_annotations, temp_df], axis=0)

    # create partitions for model training
    train_df, test_df, _, __ = train_test_split(
        aggregated_annotations,
        aggregated_annotations.source,
        test_size=0.20,
        random_state=42,
        stratify=aggregated_annotations.source,
    )

    save_dir = Path("data", "processed")
    save_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(Path("data", "processed", "train_data.parquet"), engine='pyarrow')
    test_df.to_parquet(Path("data", "processed", "test_data.parquet"), engine='pyarrow')
