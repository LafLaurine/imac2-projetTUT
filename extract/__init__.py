
def __init__():
    ### DEFAULT CONFIGURATION ###
    ## Face detection model (can't touch this

    dir_model_face_default = "face_model"
    config_face_default         = dir_model_face_default + os.sep + "deploy.prototxt.txt"
    model_face_default          = dir_model_face_default + os.sep + "res10_300x300_ssd_iter_140000.caffemodel"
    size_net_default                 = 300
    mean_default                     = (104.0, 177.0, 123.0)
    ## Feature warping model
    dir_model_landmark_default = "landmark_model"
    model_landmark_default            = dir_model_landmark_default + os.sep + "lbfmodel.yaml"


    ## Detection parameters
    method_detection_default         = DetectionMethod.dnn_tracking
    type_tracker_default             = TrackerType.csrt #most accurate, quite slow
    rate_enlarge_default             = 0.90
    min_confidence_default           = 0.95
    step_frame_default               = 1

    ##Feature warping parameters
    pair_left_eye_default            = (0.66, 0.4)
    pair_right_eye_default           = (0.33, 0.4) #IN [0, 1], proportion of face image dimensions
    pair_mouth_default               = (0.5, 0.75)
    pairs_interest_prop_default      = (pair_left_eye_default,
                                   pair_right_eye_default,
                                   pair_mouth_default)
    are_warped_default                 = True
    are_culled_default                 = True

    """
    Border mode  
    """
    mode_border_default              = cv2.BORDER_REFLECT
    method_resize_default            = cv2.INTER_LINEAR
    pair_resize_default              = (300, 300)

    are_saved_default                 = False
    log_enabled_default              = True