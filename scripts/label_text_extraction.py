"""Module to extract labels from bounding boxes."""
import json
import re

from pathlib import Path
from typing import Dict
from copy import deepcopy

import pandas as pd

from joblib import Parallel, delayed
from joblib import parallel_backend


class AnnotationsProcessor:
    def __init__(self, overlap_ratio=0.3) -> None:
        self.overlap_ratio = overlap_ratio
        self.annotation_max_range = 100

    def __call__(self, annotation: Dict) -> Dict:
        if annotation["annotations"][0]["result"]:
            annotation_scaled = self._process_annotation(annotation)
            ocr_data, file_name = self._load_ocr_data(annotation["file_upload"])
            annotation_processed = self._get_text_from_ocr_df(annotation_scaled, ocr_data)
            self._save_processed_annotation(file_name, annotation_processed)

    @staticmethod
    def _load_ocr_data(annotation_file_path):
        file_stem = re.findall(r"-(.*)", Path(annotation_file_path).stem)[0]
        ocr_file_path = Path("data/ocr", f"{file_stem}.json")
        file = open(ocr_file_path)
        ocr_data = json.load(file)
        return ocr_data, file_stem

    @staticmethod
    def _save_processed_annotation(file_name, annot_data):
        source_dir = Path("data/annotations")
        source_dir.mkdir(parents=True, exist_ok=True)
        source_file = Path(source_dir, f"{file_name}.json")

        with open(source_file, "w") as file:
            json.dump(annot_data, file)

    def _process_annotation(self, annotations):
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

    def _get_text_from_ocr_df(self, annotation: Dict, ocr: Dict) -> Dict:
        """Add text to annotation."""
        ocr_df = pd.DataFrame(ocr)
        ocr_df["right"] = ocr_df["left"] + ocr_df["width"]
        ocr_df["bottom"] = ocr_df["top"] + ocr_df["height"]

        for bb in annotation:
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

                text = []
                words = []
                for _, region in overlapping_words.iterrows():
                    text.append(region["text"])
                    words.append(
                        {
                            "word": region["text"],
                            "top": region["top"],
                            "left": region["left"],
                            "right": region["right"],
                            "bottom": region["bottom"],
                            "label": bb["label"],
                        }
                    )

                bb["text"] = " ".join(text)
                bb["words"] = words

            # No overlapping between labelling boxes and ocr_df.
            else:
                bb["text"] = ""
                bb["words"] = []

        return annotation


if __name__ == "__main__":

    # Load annotations data
    file = open("./annotations/project-2-at-2023-04-06-16-59-565329e3.json")
    annotations = json.load(file)

    ap = AnnotationsProcessor()
    ap(annotations[0])

    with parallel_backend('threading', n_jobs=-1):
        Parallel(n_jobs=-1)(delayed(ap)(file) for file in annotations)
