import os

from .module import df_learning as lrn
from .module import df_test as tst
from .module import common_classifier as clf
from keras.preprocessing.image import ImageDataGenerator

from ..common_config import  DIM_INPUT

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

PATH_DIR_CLASSIFIER = os.path.dirname(os.path.realpath(__file__))
# DEFAULT CONFIG
# Weights

step_save_weights_temp_default = 5

# Training
learning_rate_default = 0.00001
dl_rate_default = 1

number_epochs_default = 10
batch_size_test_default = 10
batch_size_learning_default = 10
#
batch_size_analysis_default = 10

rescale_default = 1 / 255

def print_info():
    print('\n---------- MesoNet -----------')
    print('Darius Afchar     (ENPC)')
    print('Vincent Nozick    (UPEM-LIGM)')
    print('Junichi Yamagishi (NII)')
    print('Isao Echizen      (NII)')
    print('------------------------------\n')

print_info()



def learn_from_dir(name_classifier,
                   dir_dataset,
                   batch_size=batch_size_learning_default,
                   number_epochs=number_epochs_default,
                   learning_rate=learning_rate_default,
                   step_save_weights_temp=step_save_weights_temp_default,
                   target_size=DIM_INPUT,
                   rescale=rescale_default
                   ):
    data_generator_training, data_generator_validation = lrn.load_data_generators_learning(rescale)
    classifier = clf.ClassifierLoader.get_classifier(name_classifier,
                                                     path_dir_classifier=PATH_DIR_CLASSIFIER,
                                                     learning_rate=learning_rate,
                                                     name_weights=None)
    generator_training,  generator_validation = lrn.load_dataset_learning(classifier,
                                                                          dir_dataset,
                                                                          data_generator_training,
                                                                          data_generator_validation,
                                                                          batch_size,
                                                                          target_size)
    evals_learning = lrn.learn_from_generator(classifier,
                                              generator_training,
                                              generator_validation,
                                              batch_size,
                                              number_epochs,
                                              step_save_weights_temp)
    return evals_learning


def test_from_dir(name_classifier,
                  dir_dataset_test,
                  batch_size=batch_size_test_default,
                  number_epochs=number_epochs_default,
                  target_size=DIM_INPUT,
                  rescale=rescale_default
                  ):
    # Flow images from directory and predict
    data_generator_test = tst.load_data_generator_test(rescale=rescale)
    classifier = clf.ClassifierLoader.get_classifier(name_classifier,
                                                     path_dir_classifier=PATH_DIR_CLASSIFIER,
                                                     learning_rate=0, # test does not allow learning from input
                                                     name_weights=None)
    generator_test = tst.load_dataset_test(classifier,
                                           dir_dataset_test,
                                           data_generator_test,
                                           batch_size,
                                           target_size)
    evals_test = tst.test_from_generator(classifier,
                                         generator_test,
                                         batch_size,
                                         number_epochs)
    return evals_test

def analyse_from_dir(name_classifier,
                     dir_input,
                     batch_size=batch_size_analysis_default,
                     target_size=DIM_INPUT,
                     rescale=rescale_default):
    data_generator_analysis = tst.load_data_generator_analysis(rescale)
    classifier = clf.ClassifierLoader.get_classifier(name_classifier,
                                                     path_dir_classifier=PATH_DIR_CLASSIFIER,
                                                     learning_rate=0,
                                                     name_weights=None)
    generator_analysis = tst.load_input_analysis(dir_input,
                                                 data_generator_analysis,
                                                 batch_size,
                                                 target_size)
    prediction = tst.analyse_from_generator(classifier,
                                               generator_analysis,
                                               batch_size)

    return prediction

