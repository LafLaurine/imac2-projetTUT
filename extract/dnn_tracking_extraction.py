from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

import common_face_extraction as fe
from common_face_extraction import log


def extract_faces_dnn_tracking(
                    src,             #path to video source for extraction
                    width_resized,           #width of extracted face
                    rate_enlarge,     #Rate to original bounding box to also be included (bigger boxes)
                    is_square,        #output face as a squared of dim width_resized x width
                    start_frame,      #Frame at which to begin extraction
                    end_frame,        #Frame at which to end
                    step_frame,       #read video every ... frames
                    max_frame,       #maximum number of frames to be read
                    min_confidence,   #confidence threshold
                    prototxt,        #path to prototxt configuration file
                    model,           #path to model
                    size_net,         #size of the processing dnn
                    mean,            #mean colour to be substracted
                    type_tracker,     #tracker type such as MIL, Boosting...
                    is_saved,         #save image in output directory
                    dir_out,          #output directory for faces
                    log_enabled       #log info
                    ):

        log(log_enabled,"[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(prototxt, model)

        log(log_enabled, "[INFO] reading video file...")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise IOError("Video could not be read at path: "+src)

        if end_frame is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            end_frame = total_frames - 1

        #FIRST PASS : finding faces to track
        log(log_enabled, "[INFO] finding faces to track...")
        list_people = []
        if(cap.isOpened() and start_frame < end_frame):
            ok, frame = fe.read_frame(cap, start_frame)
            if not ok:
                 return False
            #forward pass of blob through network, get prediction
            detections = fe.detect_faces(frame, net, size_net, mean)
            list_faces = fe.faces_from_detection(detections,
                                            width_resized,
                                            rate_enlarge,
                                            is_square,
                                            frame,
                                            start_frame,
                                            min_confidence)

            #loop over detected faces
            for face in list_faces:
                #every face belongs to a new person we'll have to track
                list_people.append(fe.Person(face, frame, type_tracker))
                log(log_enabled, "[INFO] found face at  #"+str(start_frame))

        if list_people == []:
            log(log_enabled, "[INFO] none found.")
            return []
        #Now that we have the people present at start_frame
        #We get faces when asked by tracking
        log(log_enabled, "[INFO] tracking faces...")
        k = 0
        frame_count = 0
        index_frame = start_frame
        former_frame = index_frame
        while(cap.isOpened()
                and index_frame < end_frame
                and (max_frame is None or frame_count < max_frame)
                ):
            #updating frame
            index_frame = start_frame + step_frame*k
            for it_frame in range(index_frame, former_frame):
                #update trackers at every frame
                #otherwise it might not be able to find the face
                ok, frame = fe.read_frame(cap, it_frame)
                if not ok:
                    break
                for person in list_people:
                    person.update_tracker(frame, it_frame)
                    
            ok, frame = fe.read_frame(cap, index_frame)
            if not ok:
               break
            #Need to update detection too
            #So that we can find faces closest to trackers
            detections = fe.detect_faces(frame, net, size_net, mean)
            list_faces = fe.faces_from_detection(detections,
                                                     width_resized,
                                                     rate_enlarge,
                                                     is_square,
                                                     frame,
                                                     index_frame,
                                                     min_confidence)
            #updating people
            index_person=0
            for person in list_people:
                #updating person by updating tracker and face
                if person.update(list_faces, frame, index_frame):
                    #if their face was found
                    frame_count += 1
                    log(log_enabled, "[INFO] Found face belonging to #"+str(index_person)+" at frame #"+str(index_frame))
                index_person += 1
            k += 1
        #freeing context
        cv2.destroyAllWindows()
        if is_saved:
            log(log_enabled, "[INFO] saving faces in "+dir_out+"...")
            #save every face of every person
            for person in list_people:
                person.save_images(dir_out)
        log(log_enabled, "[INFO] done.")
        return list_people
