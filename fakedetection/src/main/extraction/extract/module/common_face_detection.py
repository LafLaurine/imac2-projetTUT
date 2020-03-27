import numpy as np
import cv2 #REQUIRES OpenCV 3

from . import common_utils as ut
from . import common_face as fc

from . import dnn_detection as ddet


class DetectionMethod:
    dnn          = "DNN"
    dnn_tracking = "DNN_TRACKING"
    @staticmethod
    def get_functor(method_detection):
        switch = {
            DetectionMethod.dnn           : ddet.detect_faces_dnn,
            DetectionMethod.dnn_tracking  : ddet.detect_faces_dnn_tracking
        }
        functor_detection = switch.get(method_detection, None)
        if functor_detection is None:
            raise ValueError("Detection method not recognised: " + method_detection)
        return functor_detection

    @staticmethod
    def to_track(method_detection):
        return method_detection == DetectionMethod.dnn_tracking


class DetectionFunctor:
    # __ ???
    # TODO !
    pass



def compute_detection(frame,
                      net,
                      size_net,
                      mean
                      ):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame.image(), (size_net, size_net)),
        1.0, #scalefactor
        (size_net, size_net),
        mean)
    #fo, get prediction
    net.setInput(blob)
    list_detections = net.forward()
    return list_detections

def box_from_detection(list_detections,
                       index_detection,
                       frame
                       ):
    (w, h) = frame.dim().tuple()
    list_dim = [w, h, w, h]
    list_dim = (list_detections[0, 0, index_detection, 3:7]*np.array(list_dim)).astype(int)
    box = ut.BoundingBox(list_dim[0], list_dim[1], list_dim[2], list_dim[3])
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
                        frame,
                        min_confidence
                        ):
    if not is_detection_valid(list_detections, index_detection, min_confidence):
        #then that's not a good enough face, skipping.
        return False, None
    #compute the (x, y)-coordinates of bounding box
    box = box_from_detection(list_detections, index_detection, frame)
    return True, face_from_box(box, frame)

def faces_from_detection(list_detections,
                         frame,
                         min_confidence
                         ):
    list_faces = []
    nb_detections = len(list_detections[0, 0]) # implementation defined
    for i in range(nb_detections):
        ok, face = face_from_detection(list_detections, i, frame,  min_confidence)
        if ok:
            list_faces.append(face)
    return list_faces

def face_from_box(box, frame: ut.Frame):
    # TODO: DONE HERE
    return fc.Face(ut.Frame(frame.image(), frame.index(), frame.to_search()), box)

def load_network_detection(config_detection, model_detection):
    return cv2.dnn.readNetFromCaffe(config_detection, model_detection)

