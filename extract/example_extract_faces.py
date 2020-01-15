import os
import sys

from face_extraction import FaceExtractor


netSize = 300 #nodes for each layer
prototxt = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"

#src= "1.mp4"
isSaved = True
#outDir = os.path.abspath("out_faces")
extractionMethod='DNN'
logEnabled = True 

frameStart = 44
frameStep = 10

if __name__ == "__main__" :
    if len(sys.argv) != 3:
        print("usage : $ extract_faces.py [PATH_TO_VID] [OUTPUT_DIR]")
    else :
        src = sys.argv[1]
        outDir = sys.argv[2]
        FaceExtractor.extractFaces(
            src=src,
            isSaved=isSaved,
            outDir=outDir,
            extractionMethod=extractionMethod,
            frameStart=frameStart,
            frameStep=frameStep,
            logEnabled=logEnabled
            )

