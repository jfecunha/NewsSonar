"""Module that holds postprocessing methods."""
import pandas as pd


def unnormalize_box(bbox, width=2000, height=2000):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def process_predictions(predictions, id2label):
    return [
        {
            "left": pred[0][0],
            "top": pred[0][1],
            "right": pred[0][2],
            "bottom": pred[0][3],
            "width": pred[0][2] - pred[0][0],
            "height": pred[0][3] - pred[0][1],
            "label": id2label.get(pred[1]),
        }
        for idx, pred in enumerate(predictions)
    ]


def process_text_bboxes(bboxes_df):

    bboxes_df = bboxes_df.copy()
    bboxes_df["text"] = bboxes_df.text.str.strip()
    predicted_text = pd.merge(
        bboxes_df,
        pd.DataFrame(bboxes_df["bboxes"].to_list(), columns=["left", "top", "right", "bottom"], index=bboxes_df.index),
        left_index=True,
        right_index=True
        ).drop(columns='bboxes').drop_duplicates().dropna().reset_index(drop=True)

    return predicted_text
