import numpy as np
import imutils
import cv2 #REQUIRES OpenCV 3
import os

class Face:
    #__frameIndex            : frame # at which the face was extracted
    #__x1, __y1, __x2, __y2      : rectangle of the face in original image
    #__isSquare              : is face saved as a square ?
    #__img                   : image data of the face (diff dimensions thant in original)
    def __init__(self, frameIndex, x1, y1, x2, y2, isSquare, img):
        self.__frameIndex = frameIndex
        self.__x1 = x1
        self.__y1 = y1
        self.__x2 = x2
        self.__y2 = y2
        self.__isSquare = isSquare
        self.__img = img
    def x1(self):
        return self.__x1
    def y1(self):
        return self.__y1
    def x2(self):
        return self.__x2
    def y2(self):
        return self.__x2
    def w(self):
        return self.__x2 - self.__x1
    def h(self):
        return self.__y2 - self.__y1
    def resizedWidth(self):
        return self.__img.shape[0]
    def boundingBox(self):
        return (self.x1(), self.y1(), self.w(), self.h())
    def frameIndex(self):
        return self.__frameIndex
    def isSquare(self):
        return self.__isSquare
    def img(self):
        return self.__img
    def saveImage(self, outDir):
        #if output directory does not exist, create it
        if not os.path.exists(outDir) :
            os.mkdir(outDir)
        elif not os.path.isdir(outDir):
            raise IOError("Given output directory is not a directory.")
        #building path to output
        path = outDir+ os.sep + str(self.frameIndex())+"_("+str(self.x1())+'x'+str(self.y1())+").jpg"
        #saving output
        cv2.imwrite(path, self.img())

class TrackerFactory :
    @staticmethod
    def createTracker(trackerType):
        switch = {
            'MIL'           : cv2.TrackerMIL_create,
            'BOOSTING'      : cv2.TrackerBoosting_create,
            'KCF'           : cv2.TrackerKCF_create,
            'TLD'           : cv2.TrackerTLD_create,
            'MEDIANFLOW'    : cv2.TrackerMedianFlow_create,
            'GOTURN'        : cv2.TrackerGOTURN_create,
            'MOSSE'         : cv2.TrackerMOSSE_create,
            'CSRT'          : cv2.TrackerCSRT_create
        }
        return switch.get(trackerType, None)()

class Person :
    #__faces                 : list of Faces belonging to Person in video
    #__tracker               : tracker bound to Person
    #
    def __init__(self, face, frame, trackerType):
        self.__faces = [face]
        #create and init tracker with first face's bounding box
        self.__tracker = TrackerFactory.createTracker(trackerType) 
        box = self.face(0).boundingBox()
        ok = self.__tracker.init(frame,box)
        if not ok:
            raise RuntimeError("Could not initialise tracker.")
        

    def faces(self):
        return self.__faces
    def face(self, index):
        return self.__faces[index]
    def nbFaces(self):
        return len(self.__faces)

    def resizedWidth(self):
        #set by first face
        return self.face(0).resizedWidth()
    def isSquare(self):
        return self.face(0).isSquare()

    def __iter__(self):
        self.__it = 0
        return self
    def __next__(self):
        if self.__it >= self.nbFaces():
            raise StopIteration
        face = self.face(self.__it)
        self.__it += 1
        return face

    def append(self, face):
        self.__faces.append(face)
    def updateTracker(self, frame, frameIndex):
        #update bounding box from tracker
        ok, box = self.__tracker.update(frame)
        box = tuple([int(x) for x in box])
        box = (box[0], box[1], box[0]+box[2], box[1]+box[3])
        #failure of tracking as False
        if not ok:
            return False, box
        return True, box

    def updateFaces(self, box, frame, frameIndex, minConfidence, net, netSize, mean):
        #LAST CHECK: we try detecting faces on the box,
        #so that we can discard bad tracker outputs 
        cropped = getCroppedFromBoundingBox(box, frame, self.isSquare())
        detections = detectFaces(cropped, net, netSize, mean)
        if len(detections) == 0:
            return False
        #get first (and only, hopefully) face found
        ok =  isFaceValidDetection(detections,
                0,
                cropped,
                frameIndex,
                minConfidence
                )
        if not ok:
            #face was not good enough, sorry face
            return False
        #add face to list
        cropped = imutils.resize(cropped, width=self.resizedWidth())
        face = Face(frameIndex, box[0], box[1], box[2], box[3], self.isSquare(), cropped)
        self.append(face)
        return True

    def update(self, frame, frameIndex, minConfidence, net, netSize, mean):
        #need to update tracker first
        ok, box = self.updateTracker(frame, frameIndex)
        if not ok:
            return False
        ok = self.updateFaces(box, frame, frameIndex, minConfidence, net, netSize, mean)
        return ok 

    def saveImages(self, outDir):
        for face in self.faces():
            face.saveImage(outDir)

def readFrame(cap, frameIndex):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex-1)
        #reading next frame
        ok, frame = cap.read()
        if not ok:
    	    #if no frame has been grabbed
            return False, None
        return True, frame

def detectFaces(frame,
                net,
                netSize,
                mean
                ):
        #get dimensions, convert to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame,(netSize, netSize)),
                    1.0, #scalefactor
                    (netSize, netSize),
                    mean)
	#forward pass of blob through network, get prediction
        net.setInput(blob)
        detections = net.forward()
        return detections

def getCroppedFromBoundingBox(box, frame, isSquare):
        #get bounding box dimensions
        (x1, y1, x2, y2) = box
        if isSquare:
            w = x2 - x1
            h = y2 - y1
            #cropping as a square
            addToSquareX = max(0, (h-w)//2)
            addToSquareY = max(0, (w-h)//2)
            x1 = x1 - addToSquareX
            x2 = x2 + addToSquareX
            y1 = y1 - addToSquareY
            y2 = y2 + addToSquareY
        return frame[y1:y2,x1:x2]

def getBoundingBoxFromDetection(detections,
                detectionIndex,
                frame
                ):
        (h, w) = frame.shape[:2]
        box = (detections[0, 0, detectionIndex, 3:7]*np.array([w, h, w, h])).astype(int)
        return box

#returns wheter the face at #index is valid
def isFaceValidDetection(detections,
                detectionIndex,
                frame,
                frameIndex,
                minConfidence
                ):
    	#extract confidence, is it greater than minimum required ?
        confidence = detections[0, 0, detectionIndex, 2]
        #filter out weak detections
        return confidence >= minConfidence

#returns whether the face at #index is valid, and Face
def getFaceFromDetection(detections,
                  detectionIndex,
                  resizedWidth,
                  isSquare,
                  frame,
                  frameIndex,
                  minConfidence
                  ):
        if not isFaceValidDetection(detections, detectionIndex, frame, frameIndex, minConfidence):
            #then that's not a good enough face, skipping.
            return False, None
        #compute the (x, y)-coordinates of bounding box
        box = getBoundingBoxFromDetection(detections, detectionIndex, frame)
        return True, getFaceFromBox(box, resizedWidth, isSquare, frame, frameIndex) 

def getFaceFromBox(box,
                    resizedWidth,
                    isSquare,
                    frame,
                    frameIndex
                    ):
        #cropping as expected
        cropped = getCroppedFromBoundingBox(box, frame, isSquare)
	#Resizing as requested
        cropped = imutils.resize(cropped, width=resizedWidth)
        face = Face(frameIndex,box[0],box[1],box[2],box[3],isSquare, cropped)
        return face

def log(logEnabled, message):
    if logEnabled:
        print(message)
