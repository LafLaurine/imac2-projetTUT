import numpy as np
import cv2 #REQUIRES OpenCV 3
import imutils
import os

class Face:
    #__frameIndex            : frame # at which the face was extracted
    #__box                   : of the face in original image
    #__isSquare              : is face saved as a square ?
    #__img                   : image data of the face (diff dimensions thant in original)
    def __init__(self, frameIndex, box, isSquare, img):
        self.__frameIndex = frameIndex
        self.__box = tuple(box) #x1,y1,x2,y2
        self.__isSquare = isSquare
        self.__img = img
    def x1(self):
        return self.__box[0]
    def y1(self):
        return self.__box[1]
    def x2(self):
        return self.__box[2]
    def y2(self):
        return self.__box[3]
    def w(self):
        return self.x2() - self.x1()
    def h(self):
        return self.y2() - self.y1()
    def resizedWidth(self):
        return self.__img.shape[0]
    def box(self):
        return self.__box
    def rectangle(self):
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
        box = self.face(0).box()
        ok = self.initTracker(frame, box)
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
        if not ok:
            return False, box
        box = tuple([int(x) for x in box])
        rect = getRectangleFromBox(box)
        #failure of tracking as False
        return True, rect

    def initTracker(self, frame, box):
        return self.__tracker.init(frame, box)

    def updateFaces(self, listFace, rect, frame, frameIndex, minConfidence, net, netSize, mean):
        #LAST PROCESS: we get the face closest to tracker from list of faces
        closestFace = None
        maxSurface = -1
        for face in listFace:
            surface =  getSurfaceIntersection(face.rectangle(), rect)
            if maxSurface -1 or surface > maxSurface :
                maxSurface = surface
                closestFace = face
        if closestFace is None:
            return False
        self.append(closestFace)
        #Since the bounding box of the face change
        #we need to reset tracker
        self.initTracker(frame, closestFace.box())
        return True 

    def update(self, listFace, frame, frameIndex, minConfidence, net, netSize, mean):
        #need to update tracker first
        ok, rect = self.updateTracker(frame, frameIndex)
        if not ok:
            return False
        ok = self.updateFaces(listFace, rect, frame, frameIndex, minConfidence, net, netSize, mean)
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

def enlargeBox(box, enlargeRate):
    (x1, y1, x2, y2) = box
    #enlarge box from centre
    dX = x2 - x1
    dY = y2 - y1
    centre = (x1 + dX//2, y1 + dY//2)
    facXEnlarge = int(dX*(1+enlargeRate/2)/2)
    facYEnlarge = int(dY*(1+enlargeRate/2)/2)
    enlarged = (centre[0] - facXEnlarge, centre[1] - facYEnlarge,
                centre[0] + facXEnlarge, centre[1] + facYEnlarge)
    return enlarged


def getBoundingBoxFromDetection(detections,
                detectionIndex,
                enlargeRate,
                frame
                ):
    (h, w) = frame.shape[:2]
    dimList = [w,h,w,h]
    box = (detections[0, 0, detectionIndex, 3:7]*np.array(dimList)).astype(int)
    enlarged = enlargeBox(box, enlargeRate)
    return enlarged

#returns whether the face at #index is valid
def isFaceDetectionValid(detections,
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
                  enlargeRate,
                  isSquare,
                  frame,
                  frameIndex,
                  minConfidence
                  ):
    if not isFaceDetectionValid(detections, detectionIndex, frame, frameIndex, minConfidence):
        #then that's not a good enough face, skipping.
        return False, None
    #compute the (x, y)-coordinates of bounding box
    box = getBoundingBoxFromDetection(detections, detectionIndex, enlargeRate, frame)
    return True, getFaceFromBox(box, resizedWidth, isSquare, frame, frameIndex) 

def getListFaceFromDetection(detections,
                resizedWidth,
                enlargeRate,
                isSquare,
                frame,
                frameIndex,
                minConfidence
                ):
    listFace = []
    for i in range(len(detections)):
        ok, face = getFaceFromDetection(detections, i, resizedWidth, enlargeRate, isSquare, frame, frameIndex, minConfidence)
        if ok:
            listFace.append(face)
    return listFace


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
    face = Face(frameIndex,box,isSquare, cropped)
    return face

def getIntersection(rectA, rectB):
    if rectA[0] > rectB[0]:
        rectA, rectB = rectB, rectA  
    (xa1, ya1, xa2, ya2) = rectA
    (xb1, yb1, xb2, yb2) = rectB
    diffPlusX = max(0, xa2-xb1)
    if ya1 <= yb1:
        diffPlusY = max(0, ya2-ya1)
        intersection = (xa1, ya1, xa1+diffPlusX, ya1+diffPlusY)
    else:
        diffPlusY = max(0, ya1-ya2)
        intersection = (xa1, yb1, xa1+diffPlusX, yb1+diffPlusY)
    return intersection

def getSurface(rect):
    (x1, y1, x2, y2) = rect
    return (x2-x1)*(y2-y1)

def getSurfaceIntersection(rectA, rectB):
    return getSurface(getIntersection(rectA, rectB))

def getRectangleFromBox(box):
    (x1, y1, x2, y2) = box
    return (x1, y1, x2 - x1, y2 - y1)

def log(logEnabled, message):
    if logEnabled:
        print(message)
