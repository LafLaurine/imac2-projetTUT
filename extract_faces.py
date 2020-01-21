import sys
import argparse

from extract.face_extraction import FaceExtractor

# base arguments
type_tracker = "CSRT"
are_saved = True
are_warped = True
log_enabled = True

min_confidence = 0.90

rate_enlarge = 0.90 #in proportion of detected face, so that it does not crop chin and such
start_frame = 0
end_frame = None
step_frame = 1
max_frame = 40

parser = argparse.ArgumentParser(description="Extract faces and warp according to facial landmarks.")
parser.add_argument("--source", '-s', required=True, type=str, help="video from which to extract.")
parser.add_argument("--dest", '-d', required=True, type=str, help="directory in which to put extracted face.")
parser.add_argument("--method", '-m', required=True, type=str, help="""Can be either DNN or DNN_TRACKING""" )
parser.add_argument("--nowarp",'-n', action='store_true', help="Faces will not be aligned on basis of eyes and mouth." )

if __name__ == "__main__":
    args = vars(parser.parse_args())
    src = args["source"]
    dir_out = args["dest"]
    method_detection = args["method"]
    are_warped = not args["nowarp"]
    FaceExtractor.extract_faces(
        src=src,
        rate_enlarge=rate_enlarge,
        method_detection=method_detection,
        start_frame=start_frame,
        end_frame=end_frame,
        step_frame=step_frame,
        max_frame=max_frame,
        min_confidence=min_confidence,
        type_tracker=type_tracker,
        are_warped=are_warped,
        are_saved=are_saved,
        dir_out=dir_out,
        log_enabled=log_enabled
        )

