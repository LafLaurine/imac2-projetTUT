FROM ubuntu:latest
RUN apt-get update && \ 
    apt-get install python3-dev python3-pip
RUN pip3 install -U virtualenv
RUN virtualenv --system-site-packages -p python3 ./venv
RUN source ./venv/bin/activate
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install matplotlib
RUN pip install --upgrade tensorflow
RUN pip install Keras
RUN pip install black flake8 pytest pytest-cov sphinx numpydoc
RUN pip install imageio
RUN pip install face_recognition
RUN pip install imageio-ffmpeg
RUN pip install easydict
RUN pip install opencv-python
RUN pip install dlib
RUN pip install PyYAML

