import numpy as np
# Here are defined the labels given to each folder during training.
# It allows you to change the default directory name for
# a class of image.
# Note that once the classifier has been trained according to specific
# pairs (name, index), it won't work if you swap, for example,
# the indices of real and df.
####
# Just to be safe:
# This file is full of one-time global constants.
# Seriously, don't touch this
DIR_LABEL_DEEPFAKE = 'df'
DIR_LABEL_REAL = 'real'
DIR_LABEL_F2F = 'f2f'
DIR_LABEL_FACESWAP = 'faceswap'

LABEL_DEEPFAKE = (DIR_LABEL_DEEPFAKE, 0)
LABEL_REAL = (DIR_LABEL_REAL, 1)
LABEL_F2F = (DIR_LABEL_F2F, 2)
LABEL_FACESWAP = (DIR_LABEL_FACESWAP, 3)

DICT_LABELS_DF = {
    LABEL_DEEPFAKE[0]: LABEL_DEEPFAKE[1],
    LABEL_REAL[0]: LABEL_REAL[1]
}

DICT_LABELS_F2F = {
    LABEL_F2F[0]: LABEL_F2F[1],
    LABEL_REAL[0]: LABEL_REAL[1]
}


DICT_LABELS_MULTICLASS = {
    LABEL_DEEPFAKE[0]: LABEL_DEEPFAKE[1],
    LABEL_REAL[0]: LABEL_REAL[1],
    LABEL_F2F[0]: LABEL_F2F[1],
    LABEL_FACESWAP[0]: LABEL_FACESWAP[1],
}

def get_closest_label(label):
    return int(np.round(label))

def is_predicted_wrong(label_predicted, label_actual):
    return get_closest_label(label_predicted) != get_closest_label(label_actual)

def match_labels_dict(dict_labels_found, dict_labels_expected):
    for (key, value) in dict_labels_expected.items():
        if key not in dict_labels_found.keys():
            raise IOError("Label not found: ", (key, value))
    dict_labels_found = dict_labels_expected
