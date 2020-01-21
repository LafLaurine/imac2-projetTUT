### Face extraction from video

###### Using the OpenCV CNN face detector with video input, and Facemark for 68-point landmarks detection:

##### Due to a known bug in the python binding of Facemark which has yet to be merged to master, we use Boost.Python with pyboostcvconverter to call the landmark detector from OpenCV C++.

#### Dependencies
* OpenCV, OpenCV-contrib
* numpy
* CMake, Boost, Boost.Python, X11

#### Setup

	sudo apt install pip3
	pip3 install numpy
	sudo apt install libopencv-dev build-essential cmake libgtk-3-dev libboost-all-dev
	pip3 install opencv-python opencv-contrib-python 
	

#### Usage
###### Run example
	python3 example_extract_faces.py  [-h] --source SOURCE --dest DEST --method FACE_DETECTION_METHOD
