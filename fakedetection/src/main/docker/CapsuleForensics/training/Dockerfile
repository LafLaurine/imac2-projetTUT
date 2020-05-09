FROM python:3.7-slim
WORKDIR /code
ENV FLASK_APP capsule_forensics_train.py
ENV FLASK_RUN_HOST 0.0.0.0
RUN apt-get update \
    && apt-get install --no-install-recommends -y software-properties-common libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["flask", "run","--host=0.0.0.0", "--port=5007"]
