FROM python:3.7-slim
WORKDIR /code
RUN apt-get update \
    && apt-get install --no-install-recommends -y software-properties-common libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
COPY torch-1.4.0-cp37-cp37m-manylinux1_x86_64.whl torch-1.4.0-cp37-cp37m-manylinux1_x86_64.whl
COPY requirements.txt requirements.txt
RUN pip3 install torch-1.4.0-cp37-cp37m-manylinux1_x86_64.whl
RUN pip3 install -r requirements.txt
COPY . .
