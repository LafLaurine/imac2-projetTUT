import numpy as np
import dlib

import imutils.face_utils as face_utils


def compute_landmarks_person(person,  net):
    for face in person:
        compute_landmarks_face(face, net)


def compute_landmarks_face(face, net):
    # run the network on the bounding box of the face
    dlib_box = dlib.rectangle(*face.box().tuple())
    shape = net(np.array(face.image()), dlib_box)
    face.set_landmarks_original(face_utils.shape_to_np(shape))


def load_network_landmark(model_extraction):
    # Will use frontal faces only
    net = dlib.shape_predictor(model_extraction)
    return net
