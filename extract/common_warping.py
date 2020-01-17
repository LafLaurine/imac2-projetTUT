import numpy as np
import dlib

import imutils.face_utils as face_utils


def compute_feature_person(person, net):
    list_array_features = []
    for face in person:
        list_array_features.append(compute_feature_face(face, net))
    return list_array_features


def compute_feature_face(face, net):
    # face was already extracted, so we
    # run the network on the whole image
    shape = net(np.array(face.image()), dlib.rectangle(0, 0, face.w(), face.h()))
    array_features = face_utils.shape_to_np(shape)
    return array_features


def load_network_feature(model_extraction):
    # Will use frontal faces only
    net = dlib.shape_predictor(model_extraction)
    return net
