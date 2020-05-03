import cv2
import os

from .module import common_utils as ut
from .module import common_face_detection as fdet
from .module import common_landmark_detection_alt as ldet
from .module import landmark_warping as lndm

### DEFAULT CONFIGURATION ###

## MOST IMPORTANT; dimensions of the ouput
## Needs to correspond to input for classifiers
DIM_RESIZE                       = (512, 512)

## Face detection model (can't touch this)
dir_model_face_default = "face_model"
name_config_model_face_default   = "deploy.prototxt.txt"
name_model_face_default          = "res10_300x300_ssd_iter_140000.caffemodel"
size_net_default                 = 300
mean_net_default                 = (104.0, 177.0, 123.0)
## Feature warping model
dir_model_landmark_default       = "landmark_model"
name_model_landmark_default      = "lbfmodel.yaml"
name_model_landmark_alt_default  = "shape_predictor_68_face_landmarks.dat"


## Detection parameters
method_detection_default         = "DNN_TRACKING"
type_tracker_default             = "CSRT" #most accurate, quite slow
rate_enlarge_default             = 0.90 # (Higher than that might cause the landmark detection to fail) Not for alt method
min_confidence_default           = 0.95
step_frame_default               = 1

##Feature warping parameters
pair_left_eye_default            = (0.66, 0.35)
pair_right_eye_default           = (0.33, 0.35) #IN [0, 1], proportion of face image dimensions
pair_mouth_default               = (0.5, 0.72)
pairs_interest_prop_default      = (pair_left_eye_default,
                               pair_right_eye_default,
                               pair_mouth_default)

"""
Border mode  
"""
mode_border_default              = cv2.BORDER_REFLECT
method_resize_default            = cv2.INTER_LINEAR

# options
are_saved_default                 = False
are_saved_landmarks_default       = False
is_saved_rectangle_default        = False
are_warped_default                = True
are_culled_default                = True
log_enabled_default               = True


# Using pre-trained OpenCV CNN model for face detection

# TODO: could be issues with faces too close to the edges of the image
# in case of squared output. Might be a problem.
""""
Detection method can either of:
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


class FaceExtractor:

    # TODO: move dnn parameters to a config file

    @staticmethod
    def extract_faces_from_dir(
            dir_in,  #
            ext_codec, # file extension of a valid video file
            method_detection=method_detection_default,  # name of extraction method to be used
            pair_resize=DIM_RESIZE,  # DIM of extracted face
            pairs_interest_prop=pairs_interest_prop_default,
            rate_enlarge=rate_enlarge_default,  # Rate to original bounding box to also be included (bigger boxes)

            # no start or end frame
            start_frame=0,
            end_frame=None,
            step_frame=step_frame_default,  # read video every ... frames
            max_frame=None,  # maximum number of frames to be read
            min_confidence=min_confidence_default,  # confidence threshold

            mode_border=mode_border_default,
            method_resize=method_resize_default,

            dir_model_face=dir_model_face_default,
            dir_model_landmark=dir_model_landmark_default,
            name_model_face=name_model_face_default,
            name_model_landmark=name_model_landmark_alt_default,
            name_config_model_face=name_config_model_face_default,  # path to prototxt configuration file
            size_net=size_net_default,  # size of the processing dnn
            mean_net=mean_net_default,  # mean colour to be substracted

            are_warped=are_warped_default,
            are_culled=are_culled_default,
            type_tracker=type_tracker_default,  # WHEN TRACKING: tracker type such as MIL, Boosting...
            are_saved=are_saved_default,  # save image in output directory
            are_saved_landmarks=are_saved_landmarks_default,  # write landmarks to output
            is_saved_rectangle=is_saved_rectangle_default,
            dir_out=None,  # output directory for faces
            log_enabled=log_enabled_default  # output log info
                                ):
        # first, load detection and warping models
        ut.log(log_enabled, "[INFO] loading models...")
        net_face, net_landmark = FaceExtractor.load_models(dir_model_face,
                                                           dir_model_landmark,
                                                           name_model_face,
                                                           name_model_landmark,
                                                           name_config_model_face)
        # then warper
        warper = lndm.LandmarkWarper(pair_resize,
                                     pairs_interest_prop,
                                     mode_border,
                                     method_resize)

        try:
            for path_dir, names_dirs, names_files in os.walk(dir_in):
                for name_file in names_files:
                    # checking for video input
                    if not name_file.endswith(ext_codec):
                        continue
                    path_file = os.path.join(path_dir, name_file)
                    path_file_noext = os.path.splitext(path_file)[0]
                    dir_out_frames = os.path.join(dir_out, ut.get_path_without_basedir(path_file_noext))
                    # now we extract faces from this video file
                     #####
                    # then, read frames from input video source
                    ut.log(log_enabled, "[INFO] reading video file...")
                    list_frames = FaceExtractor.read_frames(path_file,
                                                            start_frame=start_frame,
                                                            end_frame=end_frame,
                                                            step_frame=step_frame,

                                                            max_frame=max_frame,
                                                            method_detection=method_detection)
                    # then face detector
                    FaceExtractor.extract_faces_from_frames(list_frames,
                                                net_face=net_face,
                                                net_landmark=net_landmark,
                                                warper=warper,
                                                method_detection=method_detection,

                                                min_confidence=min_confidence,

                                                size_net=size_net,
                                                mean_net=mean_net,

                                                are_warped=are_warped,
                                                are_culled=are_culled,
                                                type_tracker=type_tracker,
                                                are_saved=are_saved,
                                                are_saved_landmarks=are_saved_landmarks,
                                                is_saved_rectangle=is_saved_rectangle,
                                                dir_out=dir_out_frames)
        except KeyboardInterrupt:
            pass

    @staticmethod
    def display_faces_from_capture(
                        cap,        #real-time capture input stream
                        method_detection=method_detection_default,  # name of extraction method to be used
                        pair_resize=DIM_RESIZE,  # DIM of extracted face
                        pairs_interest_prop=pairs_interest_prop_default,
                        rate_enlarge=rate_enlarge_default,  # Rate to original bounding box to also be included (bigger boxes)

                        # no start or end frame
                        step_frame=step_frame_default,  # read video every ... frames
                        min_confidence=min_confidence_default,  # confidence threshold

                        mode_border=mode_border_default,
                        method_resize=method_resize_default,

                        dir_model_face=dir_model_face_default,
                        dir_model_landmark=dir_model_landmark_default,
                        name_model_face=name_model_face_default,
                        name_model_landmark=name_model_landmark_alt_default,
                        name_config_model_face=name_config_model_face_default,  # path to prototxt configuration file
                        size_net=size_net_default,  # size of the processing dnn
                        mean_net=mean_net_default,  # mean colour to be substracted

                        type_tracker=type_tracker_default,  # WHEN TRACKING: tracker type such as MIL, Boosting...
                        log_enabled=log_enabled_default  # output log info
                        ):
        # first, load detection and warping models
        ut.log(log_enabled, "[INFO] loading models...")
        net_face, net_landmark = FaceExtractor.load_models(dir_model_face,
                                                           dir_model_landmark,
                                                           name_model_face,
                                                           name_model_landmark,
                                                           name_config_model_face)
        # then warper
        warper = lndm.LandmarkWarper(pair_resize,
                                     pairs_interest_prop,
                                     mode_border,
                                     method_resize)
        # STARTING VALUES, WILL CHANGE ACCORDING TO INPUT
        are_saved_landmarks = False
        is_saved_rectangle = False
        count_frame = 0
        framerate = cap.get(cv2.CAP_PROP_FPS)
        # MAIN LOOP #
        while cap.isOpened():
            ## interface ##
            key = chr(255 & cv2.waitKey(int(1000/framerate)))
            if key == 'l':
                are_saved_landmarks = not are_saved_landmarks
            elif key == 'r':
                is_saved_rectangle = not is_saved_rectangle
            elif key == 'a':
                min_confidence = max(0.05 + min_confidence, 1.)
            elif key == 'z':
                min_confidence = min(-0.05 + min_confidence, 0.)
            elif key == 'q':
                break
            ##  ##
            ok, frame = ut.Frame.read_next(cap, 0, True)
            count_frame += 1
            if not ok:
                break
            if count_frame % step_frame != 0:
                continue
            list_people = FaceExtractor.extract_faces_from_frames([frame],
                                                net_face=net_face,
                                                net_landmark=net_landmark,
                                                warper=warper,
                                                method_detection=method_detection,
                                                min_confidence=min_confidence,
                                                size_net=size_net,
                                                mean_net=mean_net,

                                                are_warped=False,
                                                are_culled=False,
                                                type_tracker=type_tracker,
                                                are_saved=False,
                                                are_saved_landmarks=False,
                                                is_saved_rectangle=False,
                                                log_enabled=False)
            #We display the whole image, with landmarks and face rectangles
            for person in list_people:
                person.write_face_to_frame(index=0,
                                           frame=frame,
                                           are_saved_landmarks=are_saved_landmarks,
                                           is_saved_rectangle=is_saved_rectangle)
            #frame = cv2.resize(frame, (1920, 1080))
            cv2.namedWindow("whatever", cv2.WINDOW_FREERATIO)
            cv2.setWindowProperty("whatever", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            frame.show("whatever")
        cv2.destroyAllWindows()
        return



    @staticmethod
    def extract_faces_from_frames(
                        list_frames,
                        net_face,
                        net_landmark,
                        warper,
                        method_detection=method_detection_default,  # name of extraction method to be used
                        min_confidence=min_confidence_default,  # confidence threshold

                        size_net=size_net_default,
                        mean_net=mean_net_default,

                        are_warped=are_warped_default,
                        are_culled=are_culled_default,
                        type_tracker=type_tracker_default,  # WHEN TRACKING: tracker type such as MIL, Boosting...
                        are_saved=are_saved_default,  # save image in output directory
                        are_saved_landmarks=are_saved_landmarks_default,  # write landmarks to output IF NOT WARPED
                        is_saved_rectangle=is_saved_rectangle_default,  # write face detection rectangle to output IF NOT WARPED
                        dir_out=None,  # output directory for faces
                        log_enabled=log_enabled_default  # output log info
                        ):
        # then face detections, to get the list of people
        ut.log(log_enabled, "[INFO] detecting faces...")
        list_people = FaceExtractor.detect_faces(list_frames,
                                                 method_detection,
                                                 min_confidence,
                                                 net_face,
                                                 size_net,
                                                 mean_net,
                                                 type_tracker,
                                                 log_enabled
                                                 )
        ut.log(log_enabled, "[INFO] detecting landmarks...")
        FaceExtractor.compute_landmarks_people(list_people, net_landmark)
        ut.log(log_enabled, "[INFO] warping faces...")
        FaceExtractor.warp_from_landmarks(list_people=list_people,
                                          warper=warper,
                                          are_warped=are_warped,
                                          are_culled=are_culled)
        if are_saved:
            ut.log(log_enabled, "[INFO] saving output to " + dir_out)
            FaceExtractor.save_people(list_people, dir_out, are_saved_landmarks, is_saved_rectangle)
        return list_people

    @staticmethod
    def extract_faces_from_video(
                src,  # path to video source for extraction
                method_detection        = method_detection_default,  # name of extraction method to be used
                pair_resize             = DIM_RESIZE,  # width of extracted face
                pairs_interest_prop     = pairs_interest_prop_default,
                start_frame             = 0,  # Frame at which to begin extraction
                end_frame               = None,  # Frame at which to end
                step_frame              = step_frame_default,  # read video every ... frames
                max_frame               = None,  # maximum number of frames to be read
                min_confidence          = min_confidence_default,  # confidence threshold

                mode_border             = mode_border_default,
                method_resize           = method_resize_default,

                dir_model_face          = dir_model_face_default,
                dir_model_landmark      = dir_model_landmark_default,
                name_model_face         = name_model_face_default,
                name_model_landmark     = name_model_landmark_alt_default,
                name_config_model_face  = name_config_model_face_default,  # path to prototxt configuration file
                size_net                = size_net_default,  # size of the processing dnn
                mean_net                = mean_net_default,  # mean colour to be substracted

                are_warped              = are_warped_default,
                are_culled              = are_culled_default,
                type_tracker            = type_tracker_default,  # WHEN TRACKING: tracker type such as MIL, Boosting...
                are_saved               = are_saved_default,  # save image in output directory
                are_saved_landmarks     = are_saved_landmarks_default,  # write landmarks to output IF NOT WARPED
                is_saved_rectangle      = is_saved_rectangle_default, # write face detection rectangle to output IF NOT WARPED
                dir_out                 = None,  # output directory for faces
                log_enabled             = log_enabled_default  # output log info
                ):
        #first, load detection and warping models
        ut.log(log_enabled, "[INFO] loading models...")
        net_face, net_landmark = FaceExtractor.load_models(dir_model_face,
                                                           dir_model_landmark,
                                                           name_model_face,
                                                           name_model_landmark,
                                                           name_config_model_face)



        warper = lndm.LandmarkWarper(pair_resize,
                                     pairs_interest_prop,        #then, read frames from input video source
                                     mode_border,
                                     method_resize)

        # then, read frames from input video source
        ut.log(log_enabled, "[INFO] reading video file...")
        list_frames = FaceExtractor.read_frames(src, start_frame, end_frame, step_frame, max_frame, method_detection)

        list_people = FaceExtractor.extract_faces_from_frames(list_frames,
                                    net_face=net_face,
                                    net_landmark=net_landmark,
                                    warper=warper,
                                    method_detection=method_detection,

                                    min_confidence=min_confidence,

                                    size_net=size_net,
                                    mean_net=mean_net,

                                    are_warped=are_warped,
                                    are_culled=are_culled,
                                    type_tracker=type_tracker,
                                    are_saved=are_saved,
                                    are_saved_landmarks=are_saved_landmarks,
                                    is_saved_rectangle=is_saved_rectangle,
                                    dir_out=dir_out
                                    )

        return list_people

    @staticmethod
    def load_models(dir_model_face,
                    dir_model_landmark,
                    name_model_face,
                    name_model_landmark,
                    name_config_model_face
                    ):
        path_dir = os.path.dirname(os.path.realpath(__file__))
        path_dir_model_face = os.path.join(path_dir, dir_model_face)
        path_dir_model_landmark = os.path.join(path_dir, dir_model_landmark)
        path_model_face = os.path.join(path_dir_model_face, name_model_face)
        path_model_landmark = os.path.join(path_dir_model_landmark, name_model_landmark)
        path_config_model_face = os.path.join(path_dir_model_face, name_config_model_face)
        net_face = fdet.load_network_detection(path_config_model_face, path_model_face)
        net_landmark = ldet.load_network_landmark(path_model_landmark)
        return net_face, net_landmark

    @ staticmethod
    def read_frames(src,
                    start_frame,
                    end_frame,
                    step_frame,
                    max_frame,
                    method_detection
                    ):
        to_track = fdet.DetectionMethod.to_track(method_detection)
        return ut.read_frames_from_source(src, start_frame, end_frame, step_frame, max_frame, to_track)

    @staticmethod
    def detect_faces(list_frames,
                    method_detection,
                    min_confidence,
                    net,
                    size_net,
                    mean,
                    type_tracker,
                    log_enabled
            ):
        functor_detection = fdet.DetectionMethod.get_functor(method_detection)
        if method_detection == fdet.DetectionMethod.dnn_tracking:
            return functor_detection(
                list_frames      = list_frames,
                min_confidence   = min_confidence,
                net              = net,
                size_net         = size_net,
                mean             = mean,
                type_tracker     = type_tracker,
                log_enabled      = log_enabled
                )
        else:  # no-tracking method
            return functor_detection(
                list_frames      = list_frames,
                min_confidence   = min_confidence,
                net              = net,
                size_net         = size_net,
                mean             = mean,
                log_enabled      = log_enabled
                )

    @staticmethod
    def compute_landmarks_people(list_people, net_landmark):
        for person in list_people:
            ldet.compute_landmarks_person(person, net_landmark)
        return

    @staticmethod
    def warp_from_landmarks(list_people,
                            warper,
                            are_warped,
                            are_culled,
                            ):

        for person in list_people:
            if are_culled:
                person.cull_faces()
            if are_warped:
                warper.warp_person(person)

    @staticmethod
    def save_people(list_people, dir_out, are_saved_landmarks, is_saved_rectangle):
        for index, person in enumerate(list_people):
            name = "p_{}".format(index)
            dir_out_person = os.path.join(dir_out, name)
            person.save_faces(dir_out_person, are_saved_landmarks, is_saved_rectangle)
