import os
import cv2

import common_utils as ut
from common_utils import log
import common_detection as det
import common_warping as warp

from feature_warping import FeatureWarper

from common_detection import DetectionMethod
from common_tracking import TrackerType

#INFO: using caffe model and proposed method by sr6033
# https://github.com/sr6033/face-detection-with-OpenCV-and-DNN
# for face detection in video input

#TODO: could be issues with faces too close to the edges of the image
# in case of squared output. Might be a problem.
""""
Detection method can be any of:
   'DNN'            : using DNN
   'DNN_TRACKING'   : using DNN with tracking
"""
"""Tracking type can be any of:
   'MIL'
   'BOOSTING'
   'KCF'
   'TLD'
   'MEDIANFLOW'
   'GOTURN'
   'MOSSE'
   'CSRT'
"""
### DEFAULT CONFIGURATION ###
## Face detection model (can't touch this)
dir_model_detection_default      = "detection_model"
config_detection_default         = dir_model_detection_default + os.sep + "deploy.prototxt.txt"
model_detection_default          = dir_model_detection_default + os.sep + "res10_300x300_ssd_iter_140000.caffemodel"
size_net_default                 = 300
mean_default                     = (104.0, 177.0, 123.0)
## Feature warping model
dir_model_feature_default        = "feature_model"
model_feature_default            = dir_model_feature_default + os.sep + "shape_predictor_68_face_landmarks.dat"


## Detection parameters
method_detection_default         = DetectionMethod.dnn_tracking
type_tracker_default             = TrackerType.csrt #most accurate, quite slow
rate_enlarge_default             = 0.90
min_confidence_default           = 0.95
step_frame_default               = 1

##Feature warping parameters
pair_left_eye_default            = (0.72, 0.4)
pair_right_eye_default           = (0.28, 0.4) #IN [0, 1], proportion of face image dimensions
pair_mouth_default               = (0.5, 0.75)
pairs_interest_prop_default      = (pair_left_eye_default,
                               pair_right_eye_default,
                               pair_mouth_default)

"""
Border mode  
"""
mode_border_default              = cv2.BORDER_REFLECT
method_resize_default            = cv2.INTER_LINEAR
pair_resize_default              = (300, 300)

are_saved_default                 = False
log_enabled_default              = True

class FaceExtractor:
    @staticmethod
    def extract_faces(
                src,  # path to video source for extraction
                method_detection        = method_detection_default,  # name of extraction method to be used
                pair_resize             = pair_resize_default,  # width of extracted face
                pairs_interest_prop     = pairs_interest_prop_default,
                rate_enlarge            = rate_enlarge_default,  # Rate to original bounding box to also be included (bigger boxes)
                start_frame             = 0,  # Frame at which to begin extraction
                end_frame               = None,  # Frame at which to end
                step_frame              = step_frame_default,  # read video every ... frames
                max_frame               = None,  # maximum number of frames to be read
                min_confidence          = min_confidence_default,  # confidence threshold

                mode_border             = mode_border_default,
                method_resize           = method_resize_default,

                config_detection        = config_detection_default,  # path to prototxt configuration file
                model_detection         = model_detection_default,  # path to model
                size_net                = size_net_default,  # size of the processing dnn
                mean                    = mean_default,  # mean colour to be substracted

                model_feature           = model_feature_default,

                type_tracker            = type_tracker_default,  # WHEN TRACKING: tracker type such as MIL, Boosting...
                are_saved               = are_saved_default,  # save image in output directory
                dir_out                 = None,  # output directory for faces
                log_enabled             = log_enabled_default  # ouput log info
                ):
        #first, load detection and warping models
        log(log_enabled, "[INFO] loading models...")
        net_detection = det.load_network_detection(config_detection, model_detection)
        net_feature = warp.load_network_feature(model_feature)
        #then, read frames from input video source
        log(log_enabled, "[INFO] reading video file...")
        list_frames = FaceExtractor.read_frames(src, start_frame, end_frame, step_frame, max_frame, method_detection)
        #then face detections, to get the list of people
        log(log_enabled, "[INFO] detecting faces...")
        list_people = FaceExtractor.detect_faces(list_frames,
                                                method_detection,
                                                rate_enlarge,
                                                min_confidence,
                                                net_detection,
                                                size_net,
                                                mean,
                                                type_tracker,
                                                log_enabled
                                                )
        log(log_enabled, "[INFO] warping faces...")
        FaceExtractor.warp_faces(list_people,
                                 pair_resize,
                                 pairs_interest_prop,
                                 mode_border,
                                 method_resize,
                                 net_feature
                             )
        if are_saved:
            log(log_enabled, "[INFO] saving output to " + dir_out + os.sep)
            FaceExtractor.save_people(list_people, dir_out)

        log(log_enabled, "[INFO] success.")
        return list_people

    @ staticmethod
    def read_frames(src,
                    start_frame,
                    end_frame,
                    step_frame,
                    max_frame,
                    method_detection
                    ):
        to_track = DetectionMethod.to_track(method_detection)
        return ut.read_frames_from_source(src, start_frame, end_frame, step_frame, max_frame, to_track)

    @staticmethod
    def detect_faces(list_frames,
                    method_detection,
                    rate_enlarge,
                    min_confidence,
                    net,
                    size_net,
                    mean,
                    type_tracker,
                    log_enabled
            ):
        functor_detection = DetectionMethod.get_functor(method_detection)
        if method_detection == DetectionMethod.dnn_tracking:
            return functor_detection(
                list_frames      = list_frames,
                rate_enlarge     = rate_enlarge,
                min_confidence   = min_confidence,
                net              = net,
                size_net         = size_net,
                mean             = mean,
                type_tracker     = type_tracker,
                log_enabled      = log_enabled
                )
        else: #no tracking method
            return functor_detection(
                list_frames      = list_frames,
                rate_enlarge     = rate_enlarge,
                min_confidence   = min_confidence,
                net              = net,
                size_net         = size_net,
                mean             = mean,
                log_enabled      = log_enabled
                )

    @staticmethod
    def warp_faces(list_people,
                    pair_resize,
                    pairs_interest_prop,
                    mode_border,
                    method_resize,
                    net_feature,
                    ):
        # TODO: features and warping
        warper = FeatureWarper(pair_resize,
                            pairs_interest_prop,
                            mode_border,
                            method_resize,
                            )
        # TODO: discard faces which do not display the necessary points of interest? Done.
        for person in list_people:
            warp.compute_feature_person(person, net_feature)
            warper.warp_person(person)

    @staticmethod
    def save_people(list_people, dir_out):
        # does that mean I'm a doctor now?
        for person in list_people:
            person.save_images(dir_out)

