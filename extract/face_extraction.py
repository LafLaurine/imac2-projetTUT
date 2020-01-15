import os

import common_face_extraction as fe
from dnn_extraction import extract_faces_dnn
from dnn_tracking_extraction import extract_faces_dnn_tracking



#INFO: using caffe model and proposed method by sr6033
# https://github.com/sr6033/face-detection-with-OpenCV-and-DNN
# for face extraction from video input

#TODO: could be issues with faces too close to the edges of the image
# in case of squared output. Might be a problem.

#Extraction method can be any of:
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

### DEFAULT CONFIGURATION ###
## Face detection model (can't touch this)
method_extraction_default    = 'DNN'
type_tracker_default         = 'CSRT' #most accurate, quite slow

dir_model_default            = "caffemodel"
prototxt_default             = dir_model_default + os.sep + "deploy.prototxt.txt"
model_default                = dir_model_default + os.sep + "res10_300x300_ssd_iter_140000.caffemodel"

size_net_default             = 300
mean_default                 = (104.0, 177.0, 123.0)

## Extraction parameters
width_resized_default                 = 300
rate_enlarge_default           = 0.30
min_confidence_default         = 0.95

set_frame_default=1
is_square_default = True
log_enabled_default = True


class FaceExtractor:
    @staticmethod
    def extract_faces(
                    src,                                    #path to video source for extraction
                    method_extraction =method_extraction_default,#name of extraction method to be used
                    width_resized            =width_resized_default,         #width of extracted face
                    rate_enlarge      =rate_enlarge_default,   #Rate to original bounding box to also be included (bigger boxes)
                    is_square         =is_square_default,      #output face as a squared of dim width x width
                    start_frame       =0,                    #Frame at which to begin extraction
                    end_frame         =None,                 #Frame at which to end
                    step_frame        =set_frame_default,     #read video every ... frames
                    max_frame        =None,                 #maximum number of frames to be read

                    min_confidence    =min_confidence_default, #confidence threshold
                    prototxt         =prototxt_default,      #path to prototxt configuration file
                    model            =model_default,         #path to model
                    size_net          =size_net_default,       #size of the processing dnn
                    mean             =mean_default,          #mean colour to be substracted
                    type_tracker      =type_tracker_default,   #WHEN TRACKING: tracker type such as MIL, Boosting...
                    is_saved          =is_square_default,      #save image in output directory
                    dir_out           =None,                 #output directory for faces
                    log_enabled      =log_enabled_default            #ouput log info
            ):
        functor_extraction = ExtractionMethod.get_functor(method_extraction)
        if method_extraction == 'DNN_TRACKING' :
            return functor_extraction(
                src              =src,
                width_resized    =width_resized,
                rate_enlarge     =rate_enlarge,
                is_square        =is_square,
                start_frame      =start_frame,
                end_frame        =end_frame,
                step_frame       =step_frame,
                max_frame        =max_frame,
                min_confidence   =min_confidence,
                prototxt         =prototxt,
                model            =model,
                size_net         =size_net,
                mean             =mean,
                type_tracker      =type_tracker,
                is_saved         =is_saved,
                dir_out          =dir_out,
                log_enabled      =log_enabled
                )
        else: #no tracking method
            return functor_extraction(
                src              =src,
                width_resized    =width_resized,
                rate_enlarge     =rate_enlarge,
                is_square        =is_square,
                start_frame      =start_frame,
                end_frame        =end_frame,
                step_frame       =step_frame,
                max_frame        =max_frame,
                min_confidence   =min_confidence,
                prototxt         =prototxt,
                model            =model,
                size_net         =size_net,
                mean             =mean,
                is_saved         =is_saved,
                dir_out          =dir_out,
                log_enabled      =log_enabled
                )


class ExtractionMethod:
    @staticmethod
    def get_functor(method_extraction):
        switch = {
            'DNN'           : extract_faces_dnn,
            'DNN_TRACKING'  : extract_faces_dnn_tracking
        }
        return switch.get(method_extraction, None)
