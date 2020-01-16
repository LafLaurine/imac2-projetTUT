import numpy as np
import cv2
import dlib
import imutils
import imutils.face_utils as face_utils

import common_utils as ut

landmark_ids = face_utils.FACIAL_LANDMARKS_IDXS


class Point2D:
    # __x
    # __y
    def __init__(self, x, y):
        self.__x = int(x)
        self.__y = int(y)

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def __add__(self, point):
        return Point2D(self.x() + point.x(), self.y() + point.y())

    def __sub__(self, point):
        return Point2D(self.x() - point.x(), self.y() - point.y())

    def __mul__(self, alpha):
        return Point2D(self.x() * alpha, self.y() * alpha)

    # Vec2D methodz
    def norm(self):
        return np.sqrt(self.x() ** 2 + self.y() ** 2)

    def normalised(self):
        norm = self.norm()
        if norm == 0:
            raise ValueError("Null vector cannot be normalised.")
        return self * (1 / self.norm())

    def angle_to_origin_radians(self):
        norm = self.norm()
        if norm == 0:
            raise ValueError("Null vector has no angle relative to origin.")
        n_vec = self.normalised()
        tan_theta = n_vec.y() / n_vec.x()
        return np.arctan2(tan_theta)

    def angle_radians(self, other_vec):
        return self.angle_to_origin_radians() - other_vec.angle_to_origin_radians()

    @staticmethod
    def average(list_points):
        sum = Point2D(0, 0)
        for point in list_points:
            sum += point
        return sum * (1 / len(list_points))


# Vectors will be Point2D with a few added methods
Vec2D = Point2D


class Line:
    # __origin -> Point2D
    # __direction -> Vec2D
    def __init__(self, origin, direction):
        self.__origin = origin
        self.__direction = direction

    def origin(self):
        return self.__origin

    def direction(self):
        return self.__direction

    def angle_to_origin_radians(self):
        return self.direction().angle_to_origin_radians()


# TODO: go though each face and warp it

class FaceLandmarks:
    # __face
    # __array_features
    def __init__(self, face, array_features):
        self.__face = face
        self.__array_features = array_features


def compute_extraction(face, net):
    shape = net(face.image(), face.box())
    list_extractions = face_utils.shape_to_np(shape)
    return list_extractions


def landmarks_from_extraction(list_extractions, face):
    return FaceLandmarks(face, list_extractions)


def load_network_extraction(model_extraction):
    # Will use frontal faces only
    net = dlib.shape_predictor(model_extraction)
    return net
