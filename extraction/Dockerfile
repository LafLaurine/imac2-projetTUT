FROM python:3.7-slim as base
FROM base as builder
RUN mkdir /install
WORKDIR /install
RUN apt-get update \
    && apt-get install --no-install-recommends -y software-properties-common build-essential cmake libopencv-dev python-opencv libopencv-contrib-dev libboost-all-dev libgtk-3-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
