from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

import common_face_extraction as fe
from common_face_extraction import log

def extractFacesDNN(
                    src,             #path to video source for extraction
                    width,           #width of extracted face
                    isSquare,        #output face as a squared of dim width x width
                    frameStart,      #Frame at which to begin extraction
                    frameEnd,        #Frame at which to end
                    frameStep,       #read video every ... frames
                    maxFrames,       #maximum number of frames to be read
                    minConfidence,   #confidence threshold
                    prototxt,        #path to prototxt configuration file
                    model,           #path to model
                    netSize,         #size of the processing dnn
                    mean,            #mean colour to be substracted
                    isSaved,         #are the faces to be saved ?
                    outDir,          #output dir
                    logEnabled       #log info
        ):
    
        # load model
        log(logEnabled, "[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(prototxt,model)
        #reading video input
        log(logEnabled, "[INFO] reading video file...")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise IOError("Video could not be read at path: "+src)
	

        #setting up frameEnd if necessary
        if frameEnd is None :
            totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameEnd = totalFrames - 1

        k=0
        nbFrames=0
        frameIndex = frameStart
        listFace = []
        while(cap.isOpened()
                and frameIndex < frameEnd
                and (maxFrames is None or nbFrames < maxFrames)):
            #getting right frame
            #with regards to frameStart, frameEnd and frameStep
            frameIndex = frameStart + frameStep*k
            ok, frame = fe.readFrame(cap, frameIndex)
            if not ok :
                break;
            #forward pass of blob through network, get prediction
            detections = fe.detectFaces(frame, net, netSize, mean)
            #loop over the detections
            for i in range(detections.shape[2]):
                ok, face = fe.getFaceFromDetection(detections,
                                            i,
                                            width,
                                            isSquare,
                                            frame,
                                            frameIndex,
                                            minConfidence)
                if not ok:
                    #image was not close enough to a face
                    continue
                #adding face to list, saving if asked
                listFace.append(face)
                nbFrames += 1
                log(logEnabled, "[INFO] detected face at frame #"+str(frameIndex))


            k+=1
	#freeing context
        cv2.destroyAllWindows()
        #saving images if requested
        if isSaved:
            log(logEnabled, "[INFO] saving faces in "+outDir+"...")
            for face in listFace:
                face.saveImage(outDir)

        #returning list of faces
        log(logEnabled, "[INFO] done.")
        return listFace

