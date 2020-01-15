import numpy as np
import cv2 #REQUIRES OpenCV 3
import imutils
import os

class Face:
    #frame_index            : frame # at which the face was extracted
    #box                   : of the face in original image
    #is_square              : is face saved as a square ?
    #img                   : image data of the face (diff dimensions thant in original)
    def __init__(self, index_frame, box, is_square, img):
        self.__index_frame = index_frame
        self.__box = tuple(box) #x1,y1,x2,y2
        self.__is_square = is_square
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

    def resized_width(self):
        return self.__img.shape[0]
    def box(self):
        return self.__box
    def rectangle(self):
        return (self.x1(), self.y1(), self.w(), self.h())
    def index_frame(self):
        return self.__index_frame
    def is_square(self):
        return self.__is_square
    def img(self):
        return self.__img
    def save_image(self, dir_out):
        #if output directory does not exist, create it
        if not os.path.exists(dir_out) :
            os.mkdir(dir_out)
        elif not os.path.isdir(dir_out):
            raise IOError("Given output directory is not a directory.")
        #building path to output
        path = dir_out+ os.sep + str(self.index_frame())+"_("+str(self.x1())+'x'+str(self.y1())+").jpg"
        #saving output
        cv2.imwrite(path, self.img())

class TrackerFactory :
    @staticmethod
    def create_tracker(type_tracker):
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
        return switch.get(type_tracker, None)()

class Person :
    #faces                 : list of Faces belonging to Person in video
    #tracker               : tracker bound to Person
    #
    def __init__(self, face, frame, type_tracker):
        self.__faces = [face]
        #create and init tracker with first face's bounding box
        self.__tracker = TrackerFactory.create_tracker(type_tracker)
        box = self.face(0).box()
        ok = self.init_tracker(frame, box)
        if not ok:
            raise RuntimeError("Could not initialise tracker.")
        

    def faces(self):
        return self.__faces
    def face(self, index):
        return self.__faces[index]
    def faces_count(self):
        return len(self.__faces)

    def resized_width(self):
        #set by first face
        return self.face(0).resized_width()
    def is_square(self):
        return self.face(0).is_square()

    def __iter__(self):
        self.__it = 0
        return self
    def __next__(self):
        if self.__it >= self.faces_count():
            raise StopIteration
        face = self.face(self.__it)
        self.__it += 1
        return face

    def append(self, face):
        self.__faces.append(face)
    def update_tracker(self, frame, frame_index):
        #update bounding box from tracker
        ok, box = self.__tracker.update(frame)
        if not ok:
            return False, box
        box = tuple([int(x) for x in box])
        rect = rectangle_from_box(box)
        #failure of tracking as False
        return True, rect

    def init_tracker(self, frame, box):
        return self.__tracker.init(frame, box)

    def update_faces(self, list_faces, rect, frame, index_frame):
        #LAST STEP: we get the face closest to tracker from list of faces
        face_closest = None
        max_surface = -1
        for face in list_faces:
            surface = surface_intersection(face.rectangle(), rect)
            if max_surface -1 or surface > max_surface:
                max_surface = surface
                face_closest = face
        if face_closest is None:
            return False
        self.append(face_closest)
        #Since the bounding box of the face change
        #we need to reset tracker
        self.init_tracker(frame, face_closest.box())
        return True 

    def update(self, list_faces, frame, index_frame):
        #need to update tracker first
        ok, rect = self.update_tracker(frame, index_frame)
        if not ok:
            return False
        ok = self.update_faces(list_faces, rect, frame, index_frame)
        return ok 

    def save_images(self, dir_out):
        for face in self.faces():
            face.save_image(dir_out)

def read_frame(cap, index_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame-1)
    #reading next frame
    ok, frame = cap.read()
    if not ok:
        #if no frame has been grabbed
        return False, None
    return True, frame

def detect_faces(frame,
                net,
                size_net,
                mean
                ):
    #get dimensions, convert to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
                cv2.resize(frame,(size_net, size_net)),
                1.0, #scalefactor
                (size_net, size_net),
                mean)
    #forward pass of blob through network, get prediction
    net.setInput(blob)
    detections = net.forward()
    return detections

def crop_from_box(box, frame, is_square):
    #get bounding box dimensions
    (x1, y1, x2, y2) = box
    if is_square:
        w = x2 - x1
        h = y2 - y1
        #cropping as a square
        x_offset = max(0, (h-w)//2)
        y_offset = max(0, (w-h)//2)
        x1 = x1 - x_offset
        x2 = x2 + x_offset
        y1 = y1 - y_offset
        y2 = y2 + y_offset
    return frame[y1:y2, x1:x2]

def enlarged_box(box, rate_enlarge):
    (x1, y1, x2, y2) = box
    #enlarge box from centre
    w = x2 - x1
    h = y2 - y1
    centre = (x1 + w//2, y1 + h//2)
    x_enlarge = int((w/2) * (1 + rate_enlarge/2))
    y_enlarge = int((h/2) * (1 + rate_enlarge/2))
    enlarged = (centre[0] - x_enlarge, centre[1] - y_enlarge,
                centre[0] + x_enlarge, centre[1] + y_enlarge)
    return enlarged


def box_from_detection(list_detections,
                index_detection,
                rate_enlarge,
                frame
                ):
    (h, w) = frame.shape[:2]
    list_dim = [w, h, w, h]
    box = (list_detections[0, 0, index_detection, 3:7]*np.array(list_dim)).astype(int)
    enlarged = enlarged_box(box, rate_enlarge)
    return enlarged

#returns whether the face at #index is valid
def is_face_valid(list_detections,
                index_detection,
                frame,
                index_frame,
                min_confidence
                ):
    #extract confidence, is it greater than minimum required ?
    confidence = list_detections[0, 0, index_detection, 2]
    #filter out weak detections
    return confidence >= min_confidence

#returns whether the face at #index is valid, and Face
def face_from_detection(list_detections,
                  index_detection,
                  width_resized,
                  rate_enlarge,
                  is_square,
                  frame,
                  index_frame,
                  min_confidence
                  ):
    if not is_face_valid(list_detections, index_detection, frame, index_frame, min_confidence):
        #then that's not a good enough face, skipping.
        return False, None
    #compute the (x, y)-coordinates of bounding box
    box = box_from_detection(list_detections, index_detection, rate_enlarge, frame)
    return True, face_from_box(box, width_resized, is_square, frame, index_frame)

def faces_from_detection(list_detections,
                width_resized,
                rate_enlarge,
                is_square,
                frame,
                index_frame,
                min_confidence
                ):
    list_faces = []
    for i in range(len(list_detections)):
        ok, face = face_from_detection(list_detections, i, width_resized, rate_enlarge, is_square, frame, index_frame, min_confidence)
        if ok:
            list_faces.append(face)
    return list_faces


def face_from_box(box,
                    width_resized,
                    is_square,
                    frame,
                    index_frame
                    ):
    #cropping as expected
    cropped = crop_from_box(box, frame, is_square)
    #Resizing as requested
    cropped = imutils.resize(cropped, width=width_resized)
    face = Face(index_frame,box,is_square, cropped)
    return face

def rectangle_intersection(rect_a, rect_b):
    if rect_a[0] > rect_b[0]:
        rect_a, rect_b = rect_b, rect_a
    (x1_a, y1_a, x2_a, y2_a) = rect_a
    (x1_b, y1_b, x2_b, y2_b) = rect_b
    x_inter = max(0, x2_a - x1_b)
    if y1_a <= y1_b:
        y_inter = max(0, y2_a - y1_a)
        intersection = (x1_a, y1_a, x1_a+x_inter, y1_a+y_inter)
    else:
        y_inter = max(0, y1_a-y2_a)
        intersection = (x1_a, y1_b, x1_a+x_inter, y1_b+y_inter)
    return intersection

def surface(rect):
    (x1, y1, x2, y2) = rect
    return (x2-x1)*(y2-y1)

def surface_intersection(rect_a, rect_b):
    return surface(rectangle_intersection(rect_a, rect_b))

def rectangle_from_box(box):
    (x1, y1, x2, y2) = box
    return (x1, y1, x2 - x1, y2 - y1)

def log(log_enabled, message):
    if log_enabled:
        print(message)
