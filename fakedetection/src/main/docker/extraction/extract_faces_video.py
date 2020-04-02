import json
import distutils.util
import youtube_dl
import os
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
step_frame_default = 25
max_frame_default = 50


def download_video(url,output,name,quiet=True):
    ydl_opts = {}
    ydl_opts['outtmpl'] = output + name
    ydl_opts['quiet'] = quiet
    ydl_opts['merge_output_format'] = 'mkv'
    ydl_opts['format'] = 'bestvideo+bestaudio'
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        result = ydl.extract_info(url, download=False)
        outfile = ydl.prepare_filename(result) + '.' + result['ext']
    return outfile 

@app.route('/extract_faces_video')

def extract_faces_videos() :
    if (distutils.util.strtobool(os.getenv("video_download"))):
        src = download_video(os.getenv("video_url"),'./input/',os.getenv("name_video_downloaded"))
    else:
        src = os.getenv("input_path")
        dir_out = os.getenv("output_path")
        method_detection = os.getenv("method_detection")
        start_frame = int(os.getenv("start_frame"))
        end_frame = int(os.getenv("end_frame"))
        step_frame = int(os.getenv("step_frame"))
        max_frame = int(os.getenv("max_frame"))
        are_warped = distutils.util.strtobool(os.getenv("are_warped"))
        are_culled =distutils.util.strtobool(os.getenv("are_culled"))
        are_saved_landmarks = distutils.util.strtobool(os.getenv("are_saved_landmarks"))
        is_saved_rectangle = distutils.util.strtobool(os.getenv("is_saved_rectangle"))

        FaceExtractor.extract_faces_from_video(
            src=src,
            method_detection=method_detection,
            start_frame=start_frame,
            end_frame=end_frame,
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
        s = '{"message" : "Faces from video extracted"}'
        return json.loads(s)