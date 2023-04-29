FROM python:3.9-buster

ENV PATH="/poetry/bin:$PATH" \
    POETRY_VERSION="1.4.0" \
    SERVICE_HOME="/arquivo_app" \
    PYTHONIOENCODING=utf8


WORKDIR ${SERVICE_HOME}

COPY dist/newssonar-0.2.3-py3-none-any.whl ${SERVICE_HOME}/newssonar-0.2.3-py3-none-any.whl
COPY app/service.py ${SERVICE_HOME}/service.py

RUN apt-get update \
  && apt-get -y install tesseract-ocr libtesseract-dev libtinfo5

RUN apt update \
  && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install newssonar-0.2.3-py3-none-any.whl
RUN pip install gradio==3.28.0

# Steps to install detectron
RUN pip install Pillow==9.0.0
RUN pip install numpy --upgrade
RUN pip install -q datasets seqeval pyyaml==5.1
RUN pip install -q torch==1.8.0 torchvision==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install git+https://github.com/facebookresearch/detectron2.git
# Necessary for LayoutXML tokenizer
RUN pip install sentencepiece

EXPOSE 5000

ENTRYPOINT ["python3", "service.py"]
