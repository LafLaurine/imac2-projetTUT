# MesoNet

You can find here the implementation of the network architecture and the dataset used in our paper on digital forensics. It was accepted at the [WIFS 2018 conference](http://wifs2018.comp.polyu.edu.hk).

> We present a method to automatically detect face tampering in videos. We particularly focus on two recent approaches used to generate hyper-realistic forged videos: deepfake and face2face. Traditional image forensics techniques are usually not well suited to videos due to their compression that strongly degrades the data. Thus, we follow a deep learning approach and build two networks, both with a low number of layers to focus on mesoscopic properties of the image. We evaluate those fast networks on both an existing dataset and a dataset we have constituted from online videos. Our tests demonstrate a successful detection for more than 98\% for deepfake and 95\% for face2face.

[Link to full paper](https://arxiv.org/abs/1809.00888)

[Demonstrastion video (light)](https://www.youtube.com/watch?v=vch1CmgX0LA)

## Requirements

- Python 3.5
- Numpy 1.14.2
- Keras 2.1.5

## Linux install
- sudo apt-get install python3-pip
- sudo apt-get install ffmpeg
- sudo apt-get install python3-matplotlib ??

- pip3 install tensorflow
- pip3 install keras
- pip3 install imageio
- pip3 install face_recognition
- pip3 install ffmpeg
- pip3 install matplotlib

## Usage

### Extract faces from a video
- set a number of extraction frames ("frame_subsample_count") in extract_faces.py
- rename your video "video.mp4" in the main project folder 
- run python3 extract_faces.py
- get the extracted faces in the "output" directory
- manually remove bad extractions (not corresponding to a face or not to the targeted person), as well as blured or occluded images

### Run the detection
- put your faces images in the folder "test_images" or in any of its subfolder.
- run python3 deepfakes_images
- all the images of the folders and subfolder of "test_images" will be computed
- the results are: average score (0=deepfake/f2f, 1=natural video), as well as percentage of image clissified as deepfake/f2f

### Train your dataset
- first extract the faces from many videos
- build a dataset folder "data" containing two folders "training" and "validation", both of them with subfolders "df" and "real" 
- edit "traning/df_CNN.py": datasetDir = 'data' # .i.e the folder name containing your data.
- edit line 172 to specify where to export the new weights.
- read carefully "traning/df_CNN.py" and edit if necessary.
- run python3 df_CNN.py


## Dataset

### Aligned face datasets

|Set|Size of the forged image class|Size of real image class|
|---|---|---|
|Training|5111|7250|
|Validation|2889|4259|

- Training set (~150Mo)
- Validation set (~50Mo)
- Training + Validation video set (~1.4Go)

Contact us to have access to the dataset.

## Pretrained models

You can find the pretrained weight in the `weights` folder. The `_DF` extension correspond to a model trained to classify deepfake-generated images and the `_F2F` to Face2Face-generated images.

## Help

- Download links for our dataset will be available with the [WIFS 2018](http://wifs2018.comp.polyu.edu.hk) event, but don't hesitate to contact us directly if you are already interested.

## Authors

**Darius Afchar** - École des Ponts Paristech | École Normale Supérieure (France)

**Vincent Nozick** - [Website](http://www-igm.univ-mlv.fr/~vnozick/?lang=fr)

## References

Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018, September). [MesoNet: a Compact Facial Video Forgery Detection Network](https://arxiv.org/abs/1809.00888). In IEEE Workshop on Information Forensics and Security, WIFS 2018.

This research was carried out while the authors stayed at the National Institute of Informatics, Japan.
