###Face extraction from video

######NOTICE: using https://github.com/sr6033/face-detection-with-OpenCV-and-DNN to extract faces in video input
####Dependencies
*OpenCV, OpenCV-contrib
*numpy, imutils
*CMake; Boost, Boost.Python, X11
*dlib
####Setup
	sudo apt install libopencv-dev build-essential cmake libgtk-3-dev libboost-all-dev
	sudo apt install pip3
	pip3 install numpy opencv-python opencv-contrib-python imutils scipy scikit-image dlib
	

####Usage
######Run example
	python3 example_extract_faces.py [PATH_TO_VIDEO] [OUTPUT_DIRECTORY]
