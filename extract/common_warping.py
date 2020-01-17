import numpy as np
import dlib

import imutils.face_utils as face_utils


def compute_feature_person(person, net):
    for face in person:
        compute_feature_face(face, net)


def compute_feature_face(face, net):
    # face was already extracted, so we
    # run the network on the whole cropped image
    shape = net(np.array(face.image()), dlib.rectangle(0, 0, face.w(), face.h()))
    face.set_features(face_utils.shape_to_np(shape))


def load_network_feature(model_extraction):
    # Will use frontal faces only
    net = dlib.shape_predictor(model_extraction)
    return net
