import sys

from face_extraction import FaceExtractor

# base arguments
method_detection = 'DNN_TRACKING'
type_tracker = 'MOSSE'
is_saved = True
log_enabled = True

min_confidence = 0.95

rate_enlarge = 0.20 #in [0,1]
start_frame = 44
step_frame = 1

if __name__ == "__main__" :
    if len(sys.argv) != 3:
        print("usage : $ extract_faces.py [PATH_TO_VID] [OUTPUT_DIR]")
    else:
        src = sys.argv[1]
        dir_out = sys.argv[2]
        FaceExtractor.extract_faces(
            src=src,
            rate_enlarge=rate_enlarge,
            is_saved=is_saved,
            dir_out=dir_out,
            method_detection=method_detection,
            start_frame=start_frame,
            step_frame=step_frame,
            min_confidence=min_confidence,
            type_tracker=type_tracker,
            log_enabled=log_enabled
            )

