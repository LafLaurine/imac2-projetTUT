FROM python:3.7-slim
WORKDIR /code
ENV FLASK_APP mesonet_analyse.py
ENV FLASK_RUN_HOST 0.0.0.0
RUN apt-get update \
    && apt-get install --no-install-recommends -y software-properties-common build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
COPY tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["flask", "run", "--host=0.0.0.0", "--port=5003"]