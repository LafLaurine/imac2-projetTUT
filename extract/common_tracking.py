import cv2 # Requires OpenCV 3.4

class TrackerType:
    mil           = "MIL"
    boosting      = "BOOSTING"
    kcf           = "KCF"
    tld           = "TLD"
    medianflow    = "MEDIANFLOX"
    goturn        = "GOTURN"
    mosse         = "MOSSE"
    csrt          = "CSRT"
    @staticmethod
    def create_tracker(type_tracker):
        switch = {
            TrackerType.mil           : cv2.TrackerMIL_create,
            TrackerType.boosting      : cv2.TrackerBoosting_create,
            TrackerType.kcf           : cv2.TrackerKCF_create,
            TrackerType.tld           : cv2.TrackerTLD_create,
            TrackerType.medianflow    : cv2.TrackerMedianFlow_create,
            TrackerType.goturn        : cv2.TrackerGOTURN_create,
            TrackerType.mosse         : cv2.TrackerMOSSE_create,
            TrackerType.csrt          : cv2.TrackerCSRT_create
        }
        construct_tracker = switch.get(type_tracker, None)
        if construct_tracker is None:
            raise ValueError("Tracker type not recognised: " + type_tracker)
        return construct_tracker()
