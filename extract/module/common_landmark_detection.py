import numpy as np
import cv2 # REQUIRES OpenCV >= 4.3 ? For Facemark python bindings.

from . import common_utils as ut
from . import landmark_detection as lndd

# Facemark uses a 68-point landmark detector


def compute_landmarks_person(person, list_frames, net):
    # if landmarking fails, we do not discard the corresponding face yet
    for face in person:
        whole_frame = next(frame for frame in list_frames if frame.index() == face.index_frame())
        ok = compute_landmarks_face(face, whole_frame, net)
        if not ok:
            continue# for now


def compute_landmarks_face(face, frame_whole, net):
    # Doc specifies that we need run the detection
    # on the whole input image.
    # Face holds information on the region of interest -> a cv::Rect
    # So how does that translate to python ?
    cpp_rectangle = lndd.CPPRect(*face.rectangle().tuple())
    landmarks_whole_coords = net.fit(frame_whole.image(), cpp_rectangle)
    if landmarks_whole_coords is None:
        return False
    landmarks = build_landmarks_from_coords(landmarks_whole_coords, face.rectangle())
    face.set_features(landmarks)
    return True

def load_network_landmark(model_landmark):
    # Will use frontal faces only
    # net = cv2.face.createFacemarkLBF()
    # net.loadModel(model_landmark)
    net = lndd.LandmarkExtractor(model_landmark)
    return net


def build_landmarks_from_coords(landmarks_whole_coords: np.ndarray, rect: ut.Rectangle):
    list_landmarks = landmarks_whole_coords
    for i in range(len(landmarks_whole_coords)):
        list_landmarks[i][0] -= rect.x()
        list_landmarks[i][1] -= rect.y()
    return list_landmarks