"""Coco annotation builder."""
from pathlib import Path
from PIL import Image

import io

import json
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Load annotations data
    file = open("./annotations/arquivo_news_coco_format.json")
    annotations = json.load(file)
    aggregated_annotations = pd.DataFrame()
    for idx, image_meta in enumerate(annotations["images"]):

        img_path = Path(image_meta["file_name"]).name
        newspaper_dir = img_path.split("-")[1]
        image_name = '-'.join(img_path.split('-')[1:])

        image_path  = Path(
            "data", "crop", newspaper_dir, image_name
        ).as_posix()

        # Convert image to bytes to be stores in pandas dataframe
        image = Image.open(image_path).convert("RGB")
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()

        temp_df = pd.DataFrame()
        temp_df["image_id"] = [image_meta['id']]
        temp_df["image_path"] = [image_path]
        temp_df["image"] = [img_data]
        temp_df["objects"] = [[annot for annot in annotations["annotations"] if annot["image_id"] == image_meta['id']]]
        temp_df["source"] = [newspaper_dir]
        aggregated_annotations = pd.concat([aggregated_annotations, temp_df], axis=0)

    # Create partitions for model training
    train_df, test_df, _, __ = train_test_split(
        aggregated_annotations,
        aggregated_annotations.source,
        test_size=0.20,
        random_state=42,
        stratify=aggregated_annotations.source,
    )

    save_dir = Path("data", "processed")
    save_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(Path("data", "processed", "train_data_coco.parquet"), engine='pyarrow')
    test_df.to_parquet(Path("data", "processed", "test_data_coco.parquet"), engine='pyarrow')

    # TODO: try to load directly from pandas
    dataset = load_dataset(
        "parquet",
        data_files={'train': 'train_data_coco.parquet', 'test': 'test_data_coco.parquet'},
        data_dir='./data/processed'
    )

    dataset.push_to_hub("jfecunha/arquivo_news_coco", token='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')
