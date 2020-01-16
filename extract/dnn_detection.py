import common_face_detection as fdet
from common_utils import log

def detect_faces_dnn(
                    list_frames,      #
                    rate_enlarge,     #Rate to original bounding box to also be included (bigger boxes)
                    min_confidence,   #confidence threshold
                    net,
                    size_net,         #size of the processing dnn
                    mean,             #mean colour to be substracted
                    log_enabled       #log info
        ):
        person = None
        for frame in list_frames:
            #forward pass of blob through network, get prediction
            list_detections = fdet.compute_detection(frame, net, size_net, mean)
            #IMPORTANT; since we do not track people in this method
            #we can only assume that is only one person.
            #so we only get the first result from detection

            #get first new face from detection
            ok, face = fdet.face_from_detection(list_detections,
                                            0,
                                            rate_enlarge,
                                            frame,
                                            min_confidence)
            if not ok:
                #no face in this, frame, next.
                continue
            if person is None: #can't find the joke, but I know there's one somewhere
                person = fdet.Person(face, frame, None, False) #no need to track this person
            else:
                person.append(face)
            if(len(list_detections) != 0):
                log(log_enabled, "[INFO] detected faces at frame #"+str(frame.index()))
        #returning list of peole (only one)
        if person is None:
            list_people = []
        else:
            list_people = [person]
        return list_people

