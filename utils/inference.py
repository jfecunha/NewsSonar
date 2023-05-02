"""Inference methods."""
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
import pytesseract

from utils.postprocessing import process_predictions, unnormalize_box, process_text_bboxes
from utils.preprocessing import DataExtractor
from scripts.build_dataset import DatasetBuilder

from transformers import LayoutLMv2ForTokenClassification, LayoutXLMProcessor
from datasets import Dataset

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


def make_model_request(model, processor, img, file, id2label):
    encoded_inputs = processor(
        img,
        file["words"],
        boxes=file["bboxes"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_offsets_mapping=True,
        stride=128,
    )

    offset_mapping = encoded_inputs.pop("offset_mapping")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for k, v in encoded_inputs.items():
        encoded_inputs[k] = v.to(device)

    model.to(device)

    # forward pass
    outputs = model(**encoded_inputs)
    print(outputs.logits.shape)

    # Mask to control subwords
    is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoded_inputs.bbox.squeeze().tolist()

    predictions_processed = [
        id2label[prediction]
        for idx, prediction in enumerate(predictions)
        if not is_subword[idx]
    ]
    token_boxes_processed = [
        unnormalize_box(box)
        for idx, box in enumerate(token_boxes)
        if not is_subword[idx]
    ]

    model_response = process_predictions(zip(token_boxes_processed, predictions_processed), id2label)

    return model_response, token_boxes_processed, predictions_processed


def main(img):
    img_ocr = DB_BUILDER.run_ocr(np.asarray(img))
    words = pd.DataFrame(DataExtractor.flat_list([val["words"] for val in img_ocr]))
    img_data = Dataset.from_dict(
        {
            "words": list(words["text"].values),
            "bboxes": [
                DB_BUILDER.normalize_bbox(
                    [val["left"], val["top"], val["right"], val["bottom"]], 2000, 2000
                )
                for _, val in words.iterrows()
            ],
        }
    )

    bboxes_df, bboxes, labels = make_model_request(
        model=MODEL, processor=PROCESSOR, img=img, file=img_data, id2label=ID2LABEL
    )

    # Switch to Inference mode
    DB_BUILDER.train = False
    predicted_text = DB_BUILDER._get_text_from_ocr_df(bboxes_df, img_ocr)
    predicted_text = process_text_bboxes(predicted_text)

    img_to_draw = deepcopy(img)
    draw = ImageDraw.Draw(img_to_draw)

    font = ImageFont.load_default()

    label2color = {'None':'blue', 'Category':'green', 'Title':'orange', 'SubTitle':'violet'}

    for prediction, box in zip(labels, bboxes):
        draw.rectangle(box, outline=label2color[prediction])

    return img, img_to_draw, predicted_text


if __name__ == "__main__":

    # Necessary inputs
    LABEL2ID = {"None": 0, "Title": 1, "SubTitle": 2, "Category": 3}

    ID2LABEL = {v: k for v, k in enumerate(LABEL2ID)}
    MODEL = LayoutLMv2ForTokenClassification.from_pretrained(
        'jfecunha/arquivo-layoutxml-model',
        num_labels=len(LABEL2ID)
    )
    PROCESSOR = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base", apply_ocr=False)
    DB_BUILDER = DatasetBuilder(train=True, overlap_ratio=0.3)
