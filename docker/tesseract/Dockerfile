FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SERVICE_HOME="/tesseract"

WORKDIR ${SERVICE_HOME}

COPY data/crop/cmjornal/cmjornal-20200901070349.png ${SERVICE_HOME}
COPY scripts/apply_tesseract.py ${SERVICE_HOME}

RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && apt-get install -y python3 python3-distutils python3-pip \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

RUN apt update \
  && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install pytesseract
RUN pip3 install opencv-python
RUN pip3 install pillow

ENTRYPOINT ["python3", "apply_tesseract.py"]
