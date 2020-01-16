import common_face_detection as fdet
from common_utils import log


def detect_faces_dnn_tracking(
                    list_frames,
                    rate_enlarge,     #Rate to original bounding box to also be included (bigger boxes)
                    net,
                    size_net,         #size of the processing dnn
                    min_confidence,   #confidence threshold
                    mean,            #mean colour to be substracted
                    type_tracker,     #tracker type such as MIL, Boosting...
                    log_enabled       #log info
                    ):

        #FIRST PASS : finding faces to track
        list_people = []
        if( len(list_frames) != 0):
            frame = list_frames[0]
            #forward pass of blob through network, get prediction
            detections = fdet.compute_detection(frame, net, size_net, mean)
            list_faces = fdet.faces_from_detection(detections,
                                            rate_enlarge,
                                            frame,
                                            min_confidence)
            #loop over detected faces
            for face in list_faces:
                #every face belongs to a new person we'll have to track
                list_people.append(fdet.Person(face, frame, type_tracker))
                log(log_enabled, "[INFO] found face at  #"+str(face.index_frame()))
        if list_people == []:
            log(log_enabled, "[INFO] none found.")
            return []
        #Now that we have the people present at start_frame
        #We get faces when asked by tracking
        log(log_enabled, "[INFO] tracking faces...")
        for frame in list_frames:
            #updating frame
            index_frame = frame.index()
            ###UPDATING TRACKER AT EVERY FRAME
            #otherwise it might not be able to find the face
            for person in list_people:
                person.update_tracker(frame,)
            #Need to update detection too
            #So that we can find faces closest to trackers
            detections = fdet.compute_detection(frame, net, size_net, mean)
            list_faces = fdet.faces_from_detection(detections,
                                                     rate_enlarge,
                                                     frame,
                                                     min_confidence)
            ###UPDATING PEOPLE IF FRAME IS TO BE SEARCHED
            if not frame.to_search():
                continue
            index_person=0
            for person in list_people:
                #updating person by updating tracker and face
                if person.update(list_faces, frame):
                    #if their face was found
                    log(log_enabled, "[INFO] Found face belonging to #"+str(index_person)+" at frame #"+str(index_frame))
                index_person += 1
        return list_people
