"""Module that holds postprocessing methods."""


def unnormalize_box(bbox, width=2000, height=2000):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def process_predictions(predictions):
    return [
        {
            "left": pred[0][0],
            "top": pred[0][1],
            "right": pred[0][2],
            "bottom": pred[0][3],
            "width": pred[0][2] - pred[0][0],
            "height": pred[0][3] - pred[0][1],
            "label": pred[1],
        }
        for idx, pred in enumerate(predictions)
    ]
