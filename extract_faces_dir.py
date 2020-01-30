import argparse

from extract.face_extraction import FaceExtractor

# base arguments
type_tracker = "CSRT"
are_saved = True
log_enabled = True

min_confidence = 0.85

start_frame_default = 0
end_frame_default = None

step_frame_default = 25
max_frame_default = 50

ext_codec = '.mp4'

parser = argparse.ArgumentParser(description="Extract faces and warp according to facial landmarks.")
parser.add_argument("--source",    '-s', required=True, type=str, help="directory from which to extract.")
parser.add_argument("--dest",      '-d', required=True, type=str, help="directory in which to put extracted face.")
parser.add_argument("--method",    '-m', required=True, type=str, help="""Can be either DNN or DNN_TRACKING""")
parser.add_argument("--nowarp",    '-w', action='store_true', help="Faces will not be aligned on the basis of eyes and mouth." )
parser.add_argument("--nocull",    '-c', action='store_true', help="Faces will not be culled according to out-of-bounds landmarks." )
parser.add_argument("--landmarks", '-l', action='store_true', help="Facial landmarks will be saved along with the corresponding face.")


parser.add_argument("--step",            required=False, type=int, default=step_frame_default, help="Extract faces every ... frames.")
parser.add_argument("--max",       '-m', required=False, type=int, default=max_frame_default, help="Max of frames to extract.")

if __name__ == "__main__":
    args = vars(parser.parse_args())
    dir_in =  args["source"]
    dir_out = args["dest"]
    method_detection = args["method"]
    step_frame = args["step"]
    max_frame = args["max"]
    are_warped = not args["nowarp"]
    are_culled = not args["nocull"]
    are_saved_landmarks = args["landmarks"]
    FaceExtractor.extract_faces_from_dir(
        dir_in=dir_in,
        ext_codec=ext_codec,
        method_detection=method_detection,
        step_frame=step_frame,
        max_frame=max_frame,
        min_confidence=min_confidence,
        type_tracker=type_tracker,
        are_warped=are_warped,
        are_culled=are_culled,
        are_saved=are_saved,
        are_saved_landmarks=are_saved_landmarks,
        dir_out=dir_out,
        log_enabled=log_enabled
        )

