#FROM ubuntu:18.04
FROM python:3.10-buster


ENV DEBIAN_FRONTEND=noninteractive
ENV SERVICE_HOME="/tesseract"
ENV PYTHONIOENCODING=utf8

WORKDIR ${SERVICE_HOME}

COPY data/crop/ ${SERVICE_HOME}/data/crop/
COPY scripts/apply_tesseract.py ${SERVICE_HOME}

RUN apt-get update \
  && apt-get -y install tesseract-ocr libtesseract-dev \
  && apt-get install -y python3 python3-distutils python3-pip \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*


RUN apt update \
  && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install pillow pytesseract joblib opencv-python

ENTRYPOINT ["python3", "apply_tesseract.py"]
