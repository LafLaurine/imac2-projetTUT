FROM python:3.7-slim
WORKDIR /code
ENV FLASK_APP extract_faces_video.py
ENV FLASK_RUN_HOST 0.0.0.0
RUN apt-get update \
    && apt-get install --no-install-recommends -y software-properties-common build-essential cmake libopencv-dev python-opencv libopencv-contrib-dev libboost-all-dev libgtk-3-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]