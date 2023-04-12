# NewsSonar


## Docker

```sh
docker build --rm --tag tesseract -f .\docker\tesseract\Dockerfile .
```

## Fine tuning LayoutLM on arquivo dataset

https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb#scrollTo=SedlLnHKZLAc

## Docker copy

```sh
    docker cp ocr_tessesract:/tesseract/data/ocr ./data/ocr
```

## Tesseract directory for Windows

C:\Program Files\Tesseract-OCR