from . import common_utils as ut
from . import common_face as fc
from . import common_face_detection as fdet


def detect_faces_dnn(
        list_frames,  #
        min_confidence,  # confidence threshold
        net,
        size_net,  # size of the processing dnn
        mean,  # mean colour to be substracted
        log_enabled  # log info
):
    list_people = []
    for frame in list_frames:
        # forward pass of blob through network, get prediction
        list_detections = fdet.compute_detection(frame, net, size_net, mean)
        # IMPORTANT; since we do not track people in this method
        # we can only assume that is only one person.
        list_faces = fdet.faces_from_detection(list_detections,
                                               frame,
                                               min_confidence)
        nb_faces = len(list_faces)
        if nb_faces == 0:
            continue
        #  FACES are assigned to people in the order
        # in which they appear:
        # here, a Person is only a structure to hold a list of Faces
        for i in range(nb_faces):
            face = list_faces[i]
            # If we haven't got people enough to hold as many faces in the same frame
            if i >= len(list_people):
                list_people.append(fc.Person(face, frame, None, is_tracked=False))
            else:
                list_people[i].append(face)
        """
        # so we only get the first result VALID from detection
        # get first new face VALID from detection
        face = list_faces[0]
        if person:
            # can't find the joke, but I know there's one somewhere
            # we don't track in this method
            person = fc.Person(face, frame, None, is_tracked=False)
        else:
            person.append(face)
        """
        ut.log(log_enabled, "[INFO] detected {} faces at frame #{}".format(nb_faces, frame.index()))
    # returning list of people (only one)
    """
    if person is None:
        list_people = []
        ut.log(log_enabled, "[INFO] none found.")
    else:
        list_people = [person]
    """
    return list_people


def detect_faces_dnn_tracking(
        list_frames,
        net,
        size_net,  # size of the processing dnn
        min_confidence,  # confidence threshold
        mean,  # mean colour to be substracted
        type_tracker,  # tracker type such as MIL, Boosting...
        log_enabled  # log info
):
    # FIRST PASS : finding faces to track
    list_people = []
    if (len(list_frames) != 0):
        frame = list_frames[0]
        # forward pass of blob through network, get prediction
        detections = fdet.compute_detection(frame, net, size_net, mean)
        list_faces = fdet.faces_from_detection(detections,
                                               frame,
                                               min_confidence)
        # loop over detected faces
        for face in list_faces:
            # every face belongs to a new person we'll have to track
            list_people.append(fc.Person(face, frame, type_tracker))
            ut.log(log_enabled, "[INFO] found face at  #" + str(face.index_frame()))
    if list_people == []:
        ut.log(log_enabled, "[INFO] none found.")
        return []
    # Now that we have the people present at start_frame
    # We get faces when asked by tracking
    ut.log(log_enabled, "[INFO] tracking faces...")
    for frame in list_frames:
        # updating frame
        index_frame = frame.index()
        ###UPDATING TRACKER AT EVERY FRAME
        # otherwise it might not be able to find the face
        for person in list_people:
            person.update_tracker(frame)
        # Need to update detection too
        # So that we can find faces closest to trackers
        detections = fdet.compute_detection(frame, net, size_net, mean)
        list_faces = fdet.faces_from_detection(detections,
                                               frame,
                                               min_confidence)
        ###UPDATING PEOPLE IF FRAME IS TO BE SEARCHED
        if not frame.to_search():
            continue
        index_person = 0
        for person in list_people:
            # updating person by updating tracker and face
            if person.update(list_faces, frame):
                # if their face was found
                ut.log(log_enabled,
                       "[INFO] Found face belonging to #" + str(index_person) + " at frame #" + str(index_frame))
            index_person += 1
    return list_people
