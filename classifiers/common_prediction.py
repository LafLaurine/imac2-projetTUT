import numpy as np
from . import common_labels as lab

class Prediction:
    # __dict_perc_analysis
    def __init__(self, labels_predicted, dict_labels):
        self.__dict_perc_analysis = Prediction.compute_perc_analysis(labels_predicted, dict_labels)

    @staticmethod
    def compute_perc_analysis(labels_predicted, dict_labels):
        count_labels = np.zeros(len(dict_labels))
        number_analysis = len(labels_predicted)
        for label in np.nditer(labels_predicted):
            count_labels[lab.get_closest_label(label)] += 1
        prop_labels = count_labels / number_analysis
        dict_perc_analysis = {index_label: (name_label, prop_labels[index_label])
                              for (name_label, index_label)
                              in dict_labels.items()}
        return dict_perc_analysis

    def get_prediction(self):
        this_dict = self.__dict_perc_analysis
        label, confidence = tuple(map(max, *this_dict.values()))
        return label, confidence

