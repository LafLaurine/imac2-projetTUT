import numpy as np
import pathlib

from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from .common_classifier import ImageDataGeneratorMeso, GeneratorIterationHandler
from ...common_prediction import Prediction, EvaluationTest


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
        shuffle=True,
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

def test_from_generator(classifier,
                        generator_test,
                        batch_size_test,
                        number_epochs
                        ):
    number_images = batch_size_test*number_epochs
    labels_predicted = np.ones((number_images, 1)) / 2.
    labels_actual = np.ones(number_images) / 2.
    # Predict images by batch
    for epoch in tqdm(range(number_epochs)):
        batch_images, batch_labels_actual = generator_test.next()
        batch_labels_predicted = classifier.predict(batch_images)
        #  fill the predicted and actual labels
        index_start_pred = epoch * batch_size_test
        index_end_pred = epoch * batch_size_test + len(batch_labels_predicted)
        labels_actual[index_start_pred:index_end_pred] = batch_labels_actual
        labels_predicted[index_start_pred:index_end_pred] = batch_labels_predicted
        # updating count of already processed images
    evals_test = EvaluationTest()
    evals_test.set_error_from_predicted(labels_predicted, labels_actual)
    return evals_test

def analyse_from_generator(classifier,
                           generator_analysis,
                           batch_size_analysis):

    number_images = generator_analysis.classes.shape[0]
    labels_predicted = np.ones((number_images, 1)) / 2.
    #
    count = 0
    for batch_images in tqdm(generator_analysis):
        if count * batch_size_analysis >= number_images:
            break
        batch_labels_predicted = classifier.predict(batch_images)
        ##
        index_start_pred = count * batch_size_analysis
        index_end_pred = min(count * batch_size_analysis + len(batch_labels_predicted), number_images)
        #
        labels_predicted[index_start_pred:index_end_pred] = batch_labels_predicted[0:index_end_pred - index_start_pred]
        count += 1
    prediction = Prediction(labels_predicted, classifier.get_dict_labels(), classifier.get_list_labels())
    return prediction