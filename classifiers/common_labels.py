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

DICT_LABELS = {
    DIR_LABEL_DEEPFAKE: 0,
    DIR_LABEL_REAL: 1,
    DIR_LABEL_F2F: 2,
    DIR_LABEL_FACESWAP: 3,
}


def match_labels_dict(dict_labels):
    for (key, value) in dict_labels.items():
        if key not in DICT_LABELS:
            raise IOError("Found incorrect label: ", (key, value))
        dict_labels[key] = value
