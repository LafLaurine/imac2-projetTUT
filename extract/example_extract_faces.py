import sys

from face_extraction import FaceExtractor

# base arguments
is_saved = True
method_extraction = 'DNN_TRACKING'
type_tracker = 'MOSSE'
log_enabled = True

rate_enlarge = 0.20 #in [0,1]
start_frame = 44
step_frame = 10

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
            method_extraction=method_extraction,
            start_frame=start_frame,
            step_frame=step_frame,
            type_tracker=type_tracker,
            log_enabled=log_enabled
            )

