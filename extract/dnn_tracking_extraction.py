from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


import common_face_extraction as fe
from common_face_extraction import log

def extractFacesDNNTracking(
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
                    trackerType,     #tracker type such as MIL, Boosting...
                    isSaved,         #save image in output directory
                    outDir,          #output directory for faces
                    logEnabled       #log info
                    ):

        log(logEnabled,"[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(prototxt, model)

        log(logEnabled, "[INFO] reading video file...")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise IOError("Video could not be read at path: "+src)

        if frameEnd is None:
            totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameEnd = totalFrames - 1

        #FIRST PASS : finding faces to track
        log(logEnabled, "[INFO] finding faces to track...")
        listPerson = []
        if(cap.isOpened() and frameStart < frameEnd):
            ok, frame = fe.readFrame(cap, frameStart)
            if not ok:
                 return false
            #forward pass of blob through network, get prediction
            detections = fe.detectFaces(frame, net, netSize, mean) 
            #loop over the detections
            for i in range(detections.shape[2]):
                #find each person in image
                ok, face =  fe.getFaceFromDetection(detections,
                                            i,
                                            width,
                                            isSquare,
                                            frame,
                                            frameStart,
                                            minConfidence)
                if not ok:
                    #accuracy was not satisfying, skipping
                    continue
                #This face belongs to a new person we'll have to track
                listPerson.append(fe.Person(face, frame, trackerType))
                log(logEnabled, "[INFO] found face at  #"+str(frameStart))

        if listPerson == []:
            log(logEnabled, "[INFO] none found.")
            return []
        #Now that we have the people present at frameStart
        #We get faces when asked by tracking
        log(logEnabled, "[INFO] tracking faces...")
        k=0
        frameIndex = frameStart
        formerFrameIndex = frameIndex
        while(  cap.isOpened()
                and frameIndex < frameEnd
                and (maxFrames is None or nbFrames < maxFrames)
                ):
            #updating frame
            frameIndex = frameStart + frameStep*k
            for frameIndexIt in range(frameIndex, formerFrameIndex):
                #update trackers at every frame
                #otherwise it might not be able to find the face
                ok, frame = fe.readFrame(cap, frameIndexiIt)
                if not ok:
                    break
                for person in listPerson:
                    person.updateTracker(frame, frameIndexIt)
                    
            ok, frame = fe.readFrame(cap, frameIndex)
            if not ok:
               break
            personIndex=0
            for person in listPerson:
                #updating person by updating tracker and face
                if person.update(frame, frameIndex, minConfidence, net, netSize, mean) :
                    #if their face was found
                    log(logEnabled, "[INFO] Found face belonging to #"+str(personIndex)+" at frame #"+str(frameIndex))
                personIndex+=1
            k+=1
        #freeing context
        cv2.destroyAllWindows()
        if isSaved:
            log(logEnabled, "[INFO] saving faces in "+outDir+"...")
            #save every face of every person
            for person in listPerson:
                person.saveImages(outDir)
        log(logEnabled, "[INFO] done.")
        return listPerson
