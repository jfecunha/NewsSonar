"""Module to extract labels from bounding boxes."""
from typing import Dict
from copy import deepcopy

import pandas as pd


class AnnotationsProcessor:

    def __init__(self, overlap_ratio=0.3) -> None:

        self.overlap_ratio = overlap_ratio


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def _get_text_from_ocr(self, annotation: Dict, ocr: pd.DataFrame) -> Dict:
        """Add text to annotation.

        Parameters
        ----------
        annotation : Dict
            Bounding boxes coordinates between (0-100)
        ocr : pd.DataFrame
            Tesseract OCR data.

        Returns
        -------
        Dict
            Normalized annotation with corresponding bounding boxes coordinates text.
        """

        for bb in annotation:

            # Find all OCRs overlapping with ground truth boxes.
            ocr['_l'] = ocr['left'].apply(lambda x: max(x, bb['left']))
            ocr['_t'] = ocr['top'].apply(lambda x: max(x, bb['top']))
            ocr['_r'] = ocr['right'].apply(lambda x: min(x, bb['right']))
            ocr['_b'] = ocr['bottom'].apply(lambda x: min(x, bb['bottom']))

            overlapping_words = deepcopy(ocr).query("_l < _r and _t < _b")

            if len(overlapping_words) >= 1:

                overlapping_words['int_area'] = overlapping_words.apply(
                    lambda col: (col['_r'] - col['_l']) * (col['_b'] - col['_t']), axis=1)

                overlapping_words['word_area'] = overlapping_words.apply(
                    lambda col: col['width'] * col['height'], axis=1)

                overlap_ratio = 0.3
                overlapping_words = overlapping_words.query("int_area >= @overlap_ratio * word_area")

                text = []
                words = []
                for _, region in overlapping_words.iterrows():

                    text.append(region['text'])
                    words.append(
                        {
                            'word': region['text'],
                            'top': region['top'],
                            'left': region['left'],
                            'right': region['right'],
                            'bottom': region['bottom'],
                            'label': bb['label'],
                        }
                    )

                bb['text'] = ' '.join(text)
                bb['words'] = words

            # No overlapping between labelling boxes and ocr.
            else:

                bb['text'] = ''
                bb['words'] = []

        return annotation
