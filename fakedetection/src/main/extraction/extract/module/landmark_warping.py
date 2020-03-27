import numpy as np
import cv2

from . import common_utils as ut
from . import common_face as fc



#For dlib’s 68-point facial landmark detector:
class LandmarkFinder:
    jaw             = 'jaw'
    right_eyebrow   = 'right_eyebrow'
    left_eyebrow    = 'left_eyebrow'
    nose            = 'nose'
    right_eye       = 'right_eye'
    left_eye        = 'left_eye'
    mouth           = 'mouth'
    @staticmethod
    def get_landmarks_id(name_landmark):
        switcher = {
            LandmarkFinder.jaw             : (0, 17),
            LandmarkFinder.right_eyebrow   : (17, 22),
            LandmarkFinder.left_eyebrow    : (22, 27),
            LandmarkFinder.nose            : (27, 36),
            LandmarkFinder.right_eye       : (36, 42),
            LandmarkFinder.left_eye        : (42, 48),
            LandmarkFinder.mouth           : (48, 68)
        }
        return switcher.get(name_landmark, None)


class LandmarkWarper:
    # __points_interest_prop
    # __mode_border
    # __method_resize
    # __dim_resize
    def __init__(self, pair_resize, pairs_interest_prop, mode_border, method_resize):
        self.__dim_resize = ut.Point2D(*pair_resize)
        self.__points_interest_baseline = self.__baseline_from_prop(pairs_interest_prop)
        self.__mode_border = mode_border
        self.__method_resize = method_resize

    def points_interest_baseline(self):
        return self.__points_interest_baseline

    def dim_resize(self):
        return self.__dim_resize

    def mode_border(self):
        return self.__mode_border

    def method_resize(self):
        return self.__method_resize


    def warp_person(self, person: fc.Person):
        for face in person:
            self.__warp_face(face)


    def __warp_face(self, face: fc.Face):
        """
        the three points of interest are:
        the eyes and the mouth
        """
        # first, we compute the positions of the three points of interest
        points_interest_face = LandmarkWarper.__get_points_interest(face.landmarks())
        # Then we get warp matrix from opencv FROM Face TO BASELINE
        mat_warp = self.__get_transform_matrix(points_interest_face)
        # Finally we can compute the warped image
        image_warped = self.__warp_image(face.image(), mat_warp)

        w, h = self.dim_resize().tuple()
        box_face = ut.BoundingBox(0, 0, w, h)
        landmarks_warped = LandmarkWarper.__warp_landmarks(face.landmarks(), mat_warp)
        # Face is updated with warping
        face.set_warped(box_face, image_warped, landmarks_warped)

    def __warp_image(self, image, matrix_warp):
        return cv2.warpAffine(image,
                              matrix_warp,
                              self.dim_resize().tuple(),
                              self.method_resize(),
                              self.mode_border()
                              )

    @staticmethod
    def __warp_box(box, mat_warp):
        x1, y1, x2, y2 = box.tuple()
        # perspectiveTransform expects 3D arrays
        p1 = LandmarkWarper.__warp_coords(x1, y1, mat_warp)
        p2 = LandmarkWarper.__warp_coords(x1, y2, mat_warp)
        p3 = LandmarkWarper.__warp_coords(x2, y1, mat_warp)
        p4 = LandmarkWarper.__warp_coords(x2, y2, mat_warp)
        # taking smallest box containing the warped one
        x1_warped = min(x1, p1[0], p2[0])
        y1_warped = min(y1, p1[1], p3[1])
        x2_warped = max(x2, p3[0], p4[0])
        y2_warped = max(y2, p2[1], p4[1])
        box_warped = ut.BoundingBox(x1_warped, y1_warped, x2_warped, y2_warped)
        return box_warped

    @staticmethod
    def __warp_landmarks(landmarks, mat_warp):
        landmarks_warped = np.empty_like(landmarks)
        for i in range(len(landmarks)):
            x, y = landmarks[i]
            x_warped, y_warped = LandmarkWarper.__warp_coords(x, y, mat_warp)
            landmarks_warped[i] = x_warped, y_warped
        return landmarks_warped

    @staticmethod
    def __warp_coords(x, y, mat_warp):
        # res = M * (x, y, w)
        x_warped = mat_warp[0][0]*x + mat_warp[0][1]*y + mat_warp[0][2]
        y_warped = mat_warp[1][0]*x + mat_warp[1][1]*y + mat_warp[1][2]
        return x_warped, y_warped


    def __get_transform_matrix(self, points_interest_face):
        mat_src, mat_dest = LandmarkWarper.__get_matrices_src_dest(points_interest_face,
                                                                   self.points_interest_baseline())
        return cv2.getAffineTransform(mat_src, mat_dest)

    @staticmethod
    def __get_matrices_src_dest(points_interest_face, points_interest_baseline):
        mat_src = np.float32([point.list() for point in points_interest_face])
        mat_dest = np.float32([point.list() for point in points_interest_baseline])
        return mat_src, mat_dest

    def __baseline_from_prop(self, pairs_interest_prop):
        # resulting points of interest
        list_interest_baseline = []
        w, h = self.dim_resize().tuple()
        point_dim = ut.Point2D(w, h)
        for pair_interest in pairs_interest_prop:
            point_interest = ut.Point2D(*pair_interest)
            list_interest_baseline.append(point_interest.element_wise_prod(point_dim))
        return tuple(list_interest_baseline)


    @staticmethod
    def __get_points_interest(array_features):
        return (LandmarkWarper.__get_position_left_eye(array_features),
                LandmarkWarper.__get_position_right_eye(array_features),
                LandmarkWarper.__get_position_mouth(array_features)
                )

    @staticmethod
    def __get_position_left_eye(array_features):
        pos = LandmarkWarper.__get_position_landmark_average(array_features, "left_eye")
        return pos

    @staticmethod
    def __get_position_right_eye(array_features):
        pos = LandmarkWarper.__get_position_landmark_average(array_features, "right_eye")
        return pos

    @staticmethod
    def __get_position_mouth(array_features):
        return LandmarkWarper.__get_position_landmark_average(array_features, "mouth")

    @staticmethod
    def __get_position_landmark_average(array_features, name_landmark):
        # first get list of x and y positions corresponding to landmark
        array_coords = LandmarkWarper.__get_array_coords_landmark(array_features, name_landmark)
        # build list of points
        list_points = ut.Point2D.build_from_array(array_coords)
        return ut.Point2D.average(list_points)

    @staticmethod
    def __get_array_coords_landmark(array_features, name_landmark):
        (start_index, end_index) = LandmarkFinder.get_landmarks_id(name_landmark)
        array_coords = array_features[start_index:end_index]
        return array_coords
