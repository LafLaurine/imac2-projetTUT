import numpy as np
import pathlib

from keras.preprocessing.image import ImageDataGenerator

from .common_classifier import ImageDataGeneratorMeso
from ...common_prediction import Prediction
from ...common_labels import is_predicted_wrong

def load_data_generator_test(rescale):
    generator_test = ImageDataGeneratorMeso(rescale=rescale)
    return generator_test

def load_data_generator_analysis(rescale):
    generator_analysis = ImageDataGenerator(rescale=rescale)
    return generator_analysis

def load_dataset_test(classifier,
                      dir_dataset_test,
                      data_generator_test,
                      batch_size,
                      target_size: tuple
                      ):
    generator_test = data_generator_test.flow_from_directory(
        classifier=classifier,
        directory=dir_dataset_test,
        shuffle=False,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')
    return generator_test

def load_input_analysis(dir_input,
                        data_generator_analysis,
                        batch_size,
                        target_size
                        ):
    # Silly problems require silly solutions
    # Later though I guess.
    generator_analysis = data_generator_analysis.flow_from_directory(
        directory=dir_input,
        shuffle=False,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None)
    return generator_analysis

def compute_batch_error_labels(batch_labels_predicted, batch_labels_actual):
    # for now, the error is badly computed
    batch_error_labels = np.empty_like(batch_labels_actual)
    for i in range(len(batch_labels_actual)):
        error = is_predicted_wrong(batch_labels_predicted[i], batch_labels_actual[i])
        batch_error_labels[i] = error
    return batch_error_labels

def compute_mean_squared_error(error_predicted):
    return np.mean(np.square(error_predicted))



def compute_perc_accuracy(error_prediction):
    return (1 - compute_mean_squared_error(error_prediction))*100

def test_from_generator(classifier,
                        generator_test,
                        batch_size_test,
                        ):
    # 3 - Predict images by batch
    number_images = generator_test.classes.shape[0]
    number_epochs = number_images // batch_size_test
    labels_predicted = np.ones((number_images, 1)) / 2.
    labels_actual = np.ones(number_images) / 2.
    error_predicted = np.ones(number_images)

    # every epoch
    for e in range(number_epochs + 1):
        # get image and expected label (real, false...)
        batch_images, batch_labels_actual = generator_test.next()
        print("Epoch ", e + 1, "/", number_epochs + 1)
        # get what MesoNet thinks the label is
        batch_labels_predicted = classifier.predict(batch_images)
        ##
        index_start_pred = e * batch_size_test
        index_end_pred = e * batch_size_test + len(batch_labels_predicted)
        #  fill the predicted and actual labels
        labels_predicted[index_start_pred:index_end_pred] = batch_labels_predicted
        labels_actual[index_start_pred:index_end_pred] = batch_labels_actual
        # and compute the error
        error_predicted[index_start_pred:index_end_pred] = compute_batch_error_labels(batch_labels_predicted, batch_labels_actual)
        ### ### ### ### ##
    # The following is important (mean squared)
    # We want to TEST MesoNet, we want to know how much it fails
    # Basically, we want to know the average error for real and deepfake images
    mean_squared_error = compute_mean_squared_error(error_predicted)
    perc_accuracy = compute_perc_accuracy(error_predicted)
    # The following information is actually irrelevant here
    # print('Mean prediction  :', np.mean(predicted, axis=0)[0])
    # print('Deepfake percent :', np.mean(predicted < 0.5))
    return mean_squared_error, perc_accuracy

def analyse_from_generator(classifier,
                           generator_analysis,
                           batch_size_analysis):

    number_images = generator_analysis.classes.shape[0]
    number_epochs = number_images // batch_size_analysis
    labels_predicted = np.ones((number_images, 1)) / 2.
    #
    for e in range(number_epochs + 1):
        batch_images = generator_analysis.next()
        batch_labels_predicted = classifier.predict(batch_images)
        ##
        index_start_pred = e * batch_size_analysis
        index_end_pred = e * batch_size_analysis + len(batch_labels_predicted)
        #
        labels_predicted[index_start_pred:index_end_pred] = batch_labels_predicted
    perc_analysis = Prediction(labels_predicted, classifier.get_classes())
    return perc_analysis