import os

import common_utils as ut
import common_face_detection as fdet
from dnn_detection import detect_faces_dnn
from dnn_tracking_detection import detect_faces_dnn_tracking
from common_utils import log

#INFO: using caffe model and proposed method by sr6033
# https://github.com/sr6033/face-detection-with-OpenCV-and-DNN
# for face detection in video input

#TODO: could be issues with faces too close to the edges of the image
# in case of squared output. Might be a problem.

#Detection method can be any of:
#   'DNN'            : using DNN
#   'DNN_TRACKING'   : using DNN with tracking

#Tracking type can be any of:
#   'MIL'
#   'BOOSTING'
#   'KCF'
#   'TLD'
#   'MEDIANFLOW'
#   'GOTURN'
#   'MOSSE'
#   'CSRT'

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

### DEFAULT CONFIGURATION ###
## Face detection model (can't touch this)
dir_model_detection_default  = "detection_model"
config_detection_default     = dir_model_detection_default + os.sep + "deploy.prototxt.txt"
model_detection_default      = dir_model_detection_default + os.sep + "res10_300x300_ssd_iter_140000.caffemodel"
size_net_default             = 300
mean_default                 = (104.0, 177.0, 123.0)
## Feature extraction model


## Extraction parameters
method_detection_default     = DetectionMethod.dnn_tracking
type_tracker_default         = fdet.TrackerType.csrt #most accurate, quite slow
width_resized_default        = 300
rate_enlarge_default         = 0.30
min_confidence_default       = 0.95

step_frame_default           = 1
is_square_default            = True
log_enabled_default          = True


class FaceExtractor:
    @staticmethod
    def extract_faces(
                src,  # path to video source for extraction
                method_detection        =method_detection_default,  # name of extraction method to be used
                width_resized           =width_resized_default,  # width of extracted face
                rate_enlarge            =rate_enlarge_default,  # Rate to original bounding box to also be included (bigger boxes)
                is_square               =is_square_default,  # output face as a squared of dim width x width
                start_frame             =0,  # Frame at which to begin extraction
                end_frame               =None,  # Frame at which to end
                step_frame              =step_frame_default,  # read video every ... frames
                max_frame               =None,  # maximum number of frames to be read
                min_confidence          =min_confidence_default,  # confidence threshold
                config_detection        =config_detection_default,  # path to prototxt configuration file
                model_detection         =model_detection_default,  # path to model
                size_net                =size_net_default,  # size of the processing dnn
                mean                    =mean_default,  # mean colour to be substracted
                type_tracker            =type_tracker_default,  # WHEN TRACKING: tracker type such as MIL, Boosting...
                is_saved                =is_square_default,  # save image in output directory
                dir_out                 =None,  # output directory for faces
                log_enabled             =log_enabled_default  # ouput log info
                ):
        #first, load detection model
        log(log_enabled, "[INFO] loading model...")
        net = fdet.load_network_detection(config_detection, model_detection)
        #then, read frames from input video source
        log(log_enabled, "[INFO] reading video file...")
        list_frames = FaceExtractor.read_frames(src, start_frame, end_frame, step_frame, max_frame, method_detection)
        #then face detections, and get people
        list_people = FaceExtractor.detect_faces(list_frames,
                                                method_detection,
                                                rate_enlarge,
                                                min_confidence,
                                                net,
                                                size_net,
                                                mean,
                                                type_tracker,
                                                log_enabled
                                                )
        #TODO: features and warping

        if is_saved:
            log(log_enabled, "[INFO] saving output to " + dir_out)
            for person in list_people:
                person.save_images(dir_out)

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
    def detect_faces(
                    list_frames,
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


