import numpy as np
import cv2

from imutils.face_utils import FACIAL_LANDMARKS_IDXS as landmarks_ids

import common_utils as ut
import common_detection as det


class FeatureWarper:
    # __points_interest_baseline
    # __mode_border
    # __method_resize
    # __dim_resize
    def __init__(self, pair_resize, pairs_interest_prop, mode_border, method_resize):
        self.__dim_resize = ut.Point2D(*pair_resize)
        self.__points_interest_baseline = FeatureWarper.baseline_from_prop(pairs_interest_prop, self.dim_resize())
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


    def warp_person(self, person: det.Person):
        person.cull_faces()
        for face in person:
            #self.__warp_face(face)
            face.write_features()


    def __warp_face(self, face: det.Face):
        """
        the three points of interest are:
        the eyes and the mouth
        """
        # first, we compute the positions of the three points of interest
        points_interest_face = FeatureWarper.__get_points_interest(face.features())

        # Then we get warp matrix from opencv FROM Face TO BASELINE
        matrix_warp = self.__get_warp_matrix(points_interest_face)

        # Finally we can compute the warped image
        image_warped = self.__warp_image(face.image(), matrix_warp)

        # Face is updated with warping
        face.set_image(image_warped)

    def __warp_image(self, image, matrix_warp):
        return cv2.warpAffine(image,
                              matrix_warp,
                              self.dim_resize().tuple(),
                              self.method_resize(),
                              self.mode_border()
                              )

    def __get_warp_matrix(self, points_interest_face):
        mat_src = np.float32([point.list() for point in points_interest_face])
        mat_dest = np.float32([point.list() for point in self.points_interest_baseline()])
        return cv2.getAffineTransform(mat_src, mat_dest)

    @staticmethod
    def baseline_from_prop(pairs_interest_prop, dim_resize):
        # result's points of interest
        list_interest_baseline = []
        for pair_interest in pairs_interest_prop:
            list_interest_baseline.append(ut.Point2D(*pair_interest).element_wise_prod(dim_resize))
        return tuple(list_interest_baseline)


    @staticmethod
    def __get_points_interest(array_features):
        return (FeatureWarper.__get_position_left_eye(array_features),
                FeatureWarper.__get_position_right_eye(array_features),
                FeatureWarper.__get_position_mouth(array_features)
                )

    @staticmethod
    def __get_position_left_eye(array_features):
        pos = FeatureWarper.__get_position_landmark_average(array_features, "left_eye")
        return pos

    @staticmethod
    def __get_position_right_eye(array_features):
        pos = FeatureWarper.__get_position_landmark_average(array_features, "right_eye")
        return pos

    @staticmethod
    def __get_position_mouth(array_features):
        return FeatureWarper.__get_position_landmark_average(array_features, "mouth")

    @staticmethod
    def __get_position_landmark_average(array_features, name_landmark):
        # first get list of (x, y) positions corresponding to landmark
        list_coords = FeatureWarper.__get_list_coords_landmark(array_features, name_landmark)
        # build list of points
        list_points = ut.Point2D.build_from_list(list_coords)
        return ut.Point2D.average(list_points)

    @staticmethod
    def __get_list_coords_landmark(array_features, name_landmark):
        (start_index, end_index) = landmarks_ids[name_landmark]
        list_coords = array_features[start_index:end_index]
        return list_coords
