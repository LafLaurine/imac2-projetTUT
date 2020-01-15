from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

import common_face_extraction as fe
from common_face_extraction import log


def extract_faces_dnn(
                    src,              #path to :video source for extraction
                    width_resized,    #width of extracted face
                    rate_enlarge,     #Rate to original bounding box to also be included (bigger boxes)
                    is_square,        #output face as a squared of dim width_resized x width
                    start_frame,      #Frame at which to begin extraction
                    end_frame,        #Frame at which to end
                    step_frame,       #read video every ... frames
                    max_frame,        #maximum number of frames to be read
                    min_confidence,   #confidence threshold
                    prototxt,         #path to prototxt configuration file
                    model,            #path to model
                    size_net,         #size of the processing dnn
                    mean,             #mean colour to be substracted
                    is_saved,         #are the faces to be saved ?
                    dir_out,          #output dir
                    log_enabled       #log info
        ):
    
        # load model
        log(log_enabled, "[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(prototxt,model)
        #reading video input
        log(log_enabled, "[INFO] reading video file...")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise IOError("Video could not be read at path: "+src)
        #setting up end_frame if necessary
        if end_frame is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            end_frame = total_frames - 1

        k=0
        frame_count=0
        index_frame = start_frame
        list_faces = []
        while(cap.isOpened()
                and index_frame < end_frame
                and (max_frame is None or frame_count < max_frame)):
            #getting right frame
            #with regards to start_frame, end_frame and step_frame
            index_frame = start_frame + step_frame*k
            ok, frame = fe.read_frame(cap, index_frame)
            if not ok :
                break;
            #forward pass of blob through network, get prediction
            list_detections = fe.detect_faces(frame, net, size_net, mean)
            #get new faces from list of detections
            new_faces = fe.faces_from_detection(list_detections,
                                            width_resized,
                                            rate_enlarge,
                                            is_square,
                                            frame,
                                            index_frame,
                                            min_confidence)
            list_faces += new_faces
            if(len(new_faces) != 0):
                frame_count += 1
                log(log_enabled, "[INFO] detected faces at frame #"+str(index_frame))
            k += 1
        #freeing context
        cv2.destroyAllWindows()
        #saving images if requested
        if is_saved:
            log(log_enabled, "[INFO] saving faces in "+dir_out+"...")
            for face in list_faces:
                face.save_image(dir_out)
        #returning list of faces
        log(log_enabled, "[INFO] done.")
        return list_faces

