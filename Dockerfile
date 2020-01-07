FROM ubuntu:latest
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository universe \
    && apt-get update
COPY ./MesoNet
RUN apt-get install -y cmake
RUN apt-get upgrade -y
RUN apt-get install -y python3-dev python3-pip
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install --upgrade tensorflow
RUN pip3 install Keras
RUN pip3 install black flake8 pytest pytest-cov sphinx numpydoc
RUN pip3 install imageio
RUN pip3 install face_recognition
RUN pip3 install imageio-ffmpeg
RUN pip3 install easydict
RUN pip3 install opencv-python
RUN pip3 install dlib
RUN pip3 install PyYAML
RUN pip3 freeze > requirements.txt
CMD ["python3","./MesoNet/deepfakes_images.py"]
