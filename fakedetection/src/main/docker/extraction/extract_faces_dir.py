import os
import json
import distutils.util
import redis
from flask import Flask
from extract.face_extraction import FaceExtractor

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

# base arguments
type_tracker = "CSRT"
are_saved = True
log_enabled = True

min_confidence = 0.85
start_frame_default = 0
end_frame_default = None
step_frame = 25
max_frame = 50

ext_codec = '.mp4'


@app.route('/extract_faces_dir')

def extract_faces_dir():    

    dir_in = os.getenv("input_path_dir")
    dir_out = os.getenv("output_path")
    method_detection = os.getenv("method_detection")
    are_warped = distutils.util.strtobool(os.getenv("are_warped"))
    are_culled = distutils.util.strtobool(os.getenv("are_culled"))
    are_saved_landmarks = distutils.util.strtobool(os.getenv("are_saved_landmarks"))
    is_saved_rectangle = distutils.util.strtobool(os.getenv("is_saved_rectangle"))

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
        is_saved_rectangle=is_saved_rectangle,
        dir_out=dir_out,
        log_enabled=log_enabled
        )
    s = '{"message" : "Faces from directory extracted"}'
    return json.loads(s)
