import numpy as np
import cv2 # REQUIRES OpenCV >= 4.3 ? For Facemark python bindings.

from common_utils import Rectangle
from landmark_detection import LandmarkExtractor, CPPRect

# Facemark uses a 68-point landmark detector


def compute_landmarks_person(person, list_frames, net):
    # if landmarking fails, we do not discard the corresponding face yet
    for face in person:
        whole_frame = next(frame for frame in list_frames if frame.index() == face.index_frame())
        ok = compute_landmarks_face(face, whole_frame, net)
        if not ok:
            pass # for now


def compute_landmarks_face(face, frame_whole, net):
    # Doc specifies that we need run the detection
    # on the whole input image.
    # Face holds information on the region of interest -> a cv::Rect
    # So how does that translate to python ?
    cpp_rectangle = CPPRect(*face.rectangle().tuple())
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
    net = LandmarkExtractor(model_landmark)
    return net


def build_landmarks_from_coords(landmarks_whole_coords: np.ndarray, rect: Rectangle):
    list_landmarks = []
    for i in range(len(landmarks_whole_coords)):
        x = int(landmarks_whole_coords[i][0].round()) - rect.x()
        y = int(landmarks_whole_coords[i][1].round()) - rect.y()
        list_landmarks.append((x, y))
    return list_landmarks