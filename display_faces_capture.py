import argparse

import cv2

from extract.face_extraction import FaceExtractor

# base arguments
type_tracker = "CSRT"
log_enabled = True

min_confidence = 0.85
step_frame_default = 1

parser = argparse.ArgumentParser(description="Extract faces and warp according to facial landmarks.")
parser.add_argument("--method",    '-m', required=True, type=str, help="""Can be either DNN or DNN_TRACKING""")
parser.add_argument("--step",            required=False, type=int, default=step_frame_default, help="Extract faces every ... frames.")



if __name__ == "__main__":
    args = vars(parser.parse_args())
    method_detection = args["method"]
    step_frame = args["step"]
    cap = cv2.VideoCapture(0)
    FaceExtractor.display_faces_from_capture(
        cap=cap,
        method_detection=method_detection,
        step_frame=step_frame,
        min_confidence=min_confidence,
        type_tracker=type_tracker,
        log_enabled=log_enabled
        )
    cap.release()