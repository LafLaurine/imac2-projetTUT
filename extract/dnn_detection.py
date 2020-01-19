import common_face_detection as fdet
from common_face import Face, Person

from common_utils import log



def detect_faces_dnn(
        list_frames,  #
        rate_enlarge,  # Rate to original bounding box to also be included (bigger boxes)
        min_confidence,  # confidence threshold
        net,
        size_net,  # size of the processing dnn
        mean,  # mean colour to be substracted
        log_enabled  # log info
):
    person = None
    for frame in list_frames:
        # forward pass of blob through network, get prediction
        list_detections = fdet.compute_detection(frame, net, size_net, mean)
        # IMPORTANT; since we do not track people in this method
        # we can only assume that is only one person.
        list_faces = fdet.faces_from_detection(list_detections,
                                           rate_enlarge,
                                           frame,
                                           min_confidence)
        if len(list_faces) == 0:
            continue
        # so we only get the first result VALID from detection
        # get first new face VALID from detection
        face = list_faces[0]
        if person is None:
            # can't find the joke, but I know there's one somewhere
            # we don't track in this method
            person = Person(face, frame, None, is_tracked=False)
        else:
            person.append(face)
        log(log_enabled, "[INFO] detected faces at frame #" + str(frame.index()))
    # returning list of peole (only one)
    if person is None:
        list_people = []
        log(log_enabled, "[INFO] none found.")
    else:
        list_people = [person]
    return list_people
