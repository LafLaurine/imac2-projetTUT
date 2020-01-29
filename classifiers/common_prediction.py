import numpy as np
from . import common_labels as lab

import operator as op

class Prediction:
    # __dict_prop_analysis
    def __init__(self, labels_predicted, dict_labels):
        self.__dict_prop_analysis = Prediction.compute_prop_analysis(labels_predicted, dict_labels)

    @staticmethod
    def compute_prop_analysis(labels_predicted, dict_labels):
        count_labels = np.zeros(len(dict_labels))
        number_analysis = len(labels_predicted)
        for label in np.nditer(labels_predicted):
            count_labels[lab.get_closest_label(label)] += 1
        prop_labels = count_labels / number_analysis
        dict_prop_analysis = {index_label: (name_label, prop_labels[index_label])
                              for (name_label, index_label)
                              in dict_labels.items()}
        return dict_prop_analysis

    def get_prediction(self):
        dict_values = self.__dict_prop_analysis.values()
        label, confidence = max(dict_values, key=op.itemgetter(1))
        return label, confidence

    def print(self, detailed=False):
        label, confidence = self.get_prediction()
        print("Predicted: ", label)
        print("Confidence: ", confidence)
        if detailed:
            print("Details: ")
            print(self.__dict_prop_analysis)


class EvaluationLearning:
    # __epochs[]
    # __loss_training[]
    # __acc_training[]
    # __loss_validation[]
    # __accuracy_validation[]
    def __init__(self):
        self.__epochs = []
        self.__loss_training = []
        self.__acc_training = []
        self.__loss_validation = []
        self.__acc_validation = []


    def add_eval(self, epoch, loss_training, acc_training, loss_validation, acc_validation):
        self.__epochs.append(epoch)
        self.__loss_training.append(loss_training)
        self.__loss_validation.append(loss_validation)
        self.__acc_training.append(acc_training)
        self.__acc_validation.append(acc_validation)

    def get_number_epochs(self):
        return len(self.__epochs)

    def print(self):
        for i, epoch in enumerate(self.__epochs):
            print('Epoch {0}]:'.format(epoch))
            print('    Training   -- loss: {0} | accuracy {1}'.format(self.__loss_training[i], self.__acc_training[i]))
            print('    Validation -- loss: {0} | accuracy {1}'.format(self.__loss_validation[i], self.__acc_validation[i]))

