import numpy as np
import cv2 #REQUIRES OpenCV 3

import common_utils as ut
from common_face import Face, Person


from dnn_detection import detect_faces_dnn
from dnn_tracking_detection import detect_faces_dnn_tracking


class DetectionMethod:
    dnn          = "DNN"
    dnn_tracking = "DNN_TRACKING"
    @staticmethod
    def get_functor(method_detection):
        switch = {
            DetectionMethod.dnn           : detect_faces_dnn,
            DetectionMethod.dnn_tracking  : detect_faces_dnn_tracking
        }
        functor_detection = switch.get(method_detection, None)
        if functor_detection is None:
            raise ValueError("Detection method not recognised: " + method_detection)
        return functor_detection

    @staticmethod
    def to_track(method_detection):
        return method_detection == DetectionMethod.dnn_tracking


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

