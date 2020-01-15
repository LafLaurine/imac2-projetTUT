###Face extraction from video

######NOTICE: using https://github.com/sr6033/face-detection-with-OpenCV-and-DNN to extract faces in video input
####Dependencies
*OpenCV, OpenCV-contrib
*numpy, imutils
####Setup
	sudo apt install libopencv-dev
	sudo apt install pip3
	pip3 install numpy 
	pip3 install opencv-python
	pip3 install opencv-contrib-python
	pip3 install imutils

####Usage
######Run example
	python example_extract_faces.py [PATH_TO_VIDEO] [OUTPUT_DIRECTORY]
