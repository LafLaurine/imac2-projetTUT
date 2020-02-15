import numpy as np
from . import common_config as lab

import operator as op

from .common_config import is_predicted_wrong

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

    def print(self, detailed=True):
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

    def get_loss_diff(self, order):
        return


    def print(self):
        # TODO: TO KNOW
        for i, epoch in enumerate(self.__epochs):
            print('Epoch {0}:'.format(epoch))
            print('    Training   -- loss: {0} | accuracy {1}'.format(self.__loss_training[i], self.__acc_training[i]))
            print('    Validation -- loss: {0} | accuracy {1}'.format(self.__loss_validation[i], self.__acc_validation[i]))

class EvaluationTest:
    # __array_errors
    # __mean_squared_error
    # __mean_accuracy
    def __init__(self):
        pass

    def set_error_from_predicted(self, labels_predicted, labels_actual):
        self.__array_errors = EvaluationTest.__compute_error_from_predicted(labels_predicted, labels_actual)

    def set_error(self, list_errors):
        self.__array_errors = np.array(list_errors)

    @staticmethod
    def __compute_error_from_predicted(labels_predicted, labels_actual):
        # for now, the error is badly computed
        array_errors = np.empty_like(labels_actual)
        for i in range(len(labels_actual)):
            error = is_predicted_wrong(labels_predicted[i], labels_actual[i])
            array_errors[i] = error
        return array_errors

    def get_mean_error(self):
        return np.mean(np.abs(self.__array_errors))

    def print(self):
        print('Mean error:   {}'.format(self.get_mean_error()))
