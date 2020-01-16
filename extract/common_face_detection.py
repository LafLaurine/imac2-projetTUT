import numpy as np
import cv2 #REQUIRES OpenCV 3
import os

import common_utils as ut


class TrackerType:
    mil           = "MIL"
    boosting      = "BOOSTING"
    kcf           = "KCF"
    tld           = "TLD"
    medianflow    = "MEDIANFLOX"
    goturn        = "GOTURN"
    mosse         = "MOSSE"
    csrt          = "CSRT"
    @staticmethod
    def create_tracker(type_tracker):
        switch = {
            TrackerType.mil           : cv2.TrackerMIL_create,
            TrackerType.boosting      : cv2.TrackerBoosting_create,
            TrackerType.kcf           : cv2.TrackerKCF_create,
            TrackerType.tld           : cv2.TrackerTLD_create,
            TrackerType.medianflow    : cv2.TrackerMedianFlow_create,
            TrackerType.goturn        : cv2.TrackerGOTURN_create,
            TrackerType.mosse         : cv2.TrackerMOSSE_create,
            TrackerType.csrt          : cv2.TrackerCSRT_create
        }
        construct_tracker = switch.get(type_tracker, None)
        if construct_tracker is None:
            raise ValueError("Tracker type not recognised: " + type_tracker)
        return construct_tracker()

class Face:
    #__box                   : Bounding box object of the face in original image
    #__frame                   : Frame object (w/ frame_index), diff dimensions thant original
    def __init__(self, frame, box):
        self.__frame = frame
        self.__box = box

    def x1(self):
        return self.box().x1()
    def y1(self):
        return self.box().y1()
    def x2(self):
        return self.box().x2()
    def y2(self):
        return self.box().y2()
    def w(self):
        return self.box().w()
    def h(self):
        return self.box().h()

    def box(self):
        return self.__box
    def rectangle(self):
        return ut.Rectangle.from_box(self.box())
    def index_frame(self):
        return self.__frame.index()
    def image(self):
        return self.__frame.image()

    def save_image(self, dir_out):
        #if output directory does not exist, create it
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        elif not os.path.isdir(dir_out):
            raise IOError("Given output directory is not a directory.")
        #building path to output
        filepath = dir_out + os.sep + \
               str(self.index_frame())+"_(x"+str(self.x1()) +\
               'y' + str(self.y1()) + \
               ").jpg"
        #saving output
        self.__frame.save(filepath)


class Person:
    #__faces                 : list of Faces belonging to Person in video
    #__is_tracked            ; are they tracked?
    #__tracker               : tracker bound to Person
    #
    def __init__(self, face, frame, type_tracker, is_tracked=True):
        self.__faces = [face]
        #create and init tracker with first face's bounding box
        self.__is_tracked = is_tracked
        if self.__is_tracked:
            self.__tracker = TrackerType.create_tracker(type_tracker)
            box = self.face(0).box()
            ok = self.__init_tracker(frame, box)
            if not ok:
                raise RuntimeError("Could not initialise tracker.")

    def __init_tracker(self, frame, box):
        return self.__tracker.init(frame.image(), box.tuple())

    def faces(self):
        return self.__faces
    def face(self, index):
        return self.__faces[index]
    def face_count(self):
        return len(self.__faces)

    def resized_width(self):
        #set by first face
        return self.face(0).resized_width()

    def __iter__(self):
        self.__it = 0
        return self
    def __next__(self):
        if self.__it >= self.face_count():
            raise StopIteration
        face = self.face(self.__it)
        self.__it += 1
        return face
    def append(self, face):
        self.__faces.append(face)

    def update_tracker(self, frame):
        #update bounding box from tracker
        ok, bad_box = self.__tracker.update(frame.image())
        if not ok:
            return False, None
        box = ut.BoundingBox(bad_box[0], bad_box[1], bad_box[2], bad_box[3])
        #failure of tracking as False
        return True, box

    def __update_faces(self, list_faces, box, frame):
        #LAST STEP: we get the face closest to tracker from list of faces
        rect = ut.Rectangle.from_box(box)
        face_closest = None
        max_surface = -1
        for face in list_faces:
            surface = face.rectangle().surface_intersection(rect)
            if (max_surface - 1) or (surface > max_surface):
                max_surface = surface
                face_closest = face
        if face_closest is None:
            return False
        self.append(face_closest)
        #Since the bounding box of the face changed,
        #we need to reset this person's tracker
        if self.__is_tracked:
            self.__init_tracker(frame, face_closest.box())
        return True

    def update(self, list_faces, frame):
        #need to update tracker first
        if self.__is_tracked:
            ok, box = self.update_tracker(frame)
            if not ok:
                return False
        ok = self.__update_faces(list_faces, box, frame)
        return ok

    def save_images(self, dir_out):
        for face in self.faces():
            face.save_image(dir_out)

def compute_detection(frame,
                net,
                size_net,
                mean
                ):
    #convert to a blob
    blob = cv2.dnn.blobFromImage(
                cv2.resize(frame.image(), (size_net, size_net)),
                1.0, #scalefactor
                (size_net, size_net),
                mean)
    #forward pass of blob through network, get prediction
    net.setInput(blob)
    list_detections = net.forward()
    return list_detections

def box_from_detection(list_detections,
                index_detection,
                rate_enlarge,
                frame
                ):
    (w, h) = frame.dim()
    list_dim = [w, h, w, h]
    list_dim = (list_detections[0, 0, index_detection, 3:7]*np.array(list_dim)).astype(int)
    box = ut.BoundingBox(list_dim[0], list_dim[1], list_dim[2], list_dim[3])
    box.enlarge(rate_enlarge)
    return box

#returns whether the detection at #index is valid
def is_detection_valid(list_detections,
                index_detection,
                min_confidence
                ):
    #extract confidence, is it greater than minimum required ?
    confidence = list_detections[0, 0, index_detection, 2]
    #filter out weak detections
    return confidence >= min_confidence

#returns whether the face at #index is valid, and Face
def face_from_detection(list_detections,
                  index_detection,
                  rate_enlarge,
                  frame,
                  min_confidence
                  ):
    if not is_detection_valid(list_detections, index_detection, min_confidence):
        #then that's not a good enough face, skipping.
        return False, None
    #compute the (x, y)-coordinates of bounding box
    box = box_from_detection(list_detections, index_detection, rate_enlarge, frame)
    return True, face_from_box(box, frame)

def faces_from_detection(list_detections,
                rate_enlarge,
                frame,
                min_confidence
                ):
    list_faces = []
    for i in range(len(list_detections)):
        ok, face = face_from_detection(list_detections, i, rate_enlarge, frame,  min_confidence)
        if ok:
            list_faces.append(face)
    return list_faces

def face_from_box(box, frame):
    face_image = box.crop_image(frame.image())
    return Face(ut.Frame(face_image, frame.index(), frame.to_search()), box)

def load_network_detection(config_detection, model_detection):
    return cv2.dnn.readNetFromCaffe(config_detection, model_detection)

