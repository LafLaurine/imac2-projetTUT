import os

from .module import df_learning as lrn
from .module import df_test as tst
from .module import common_classifier as clf
from keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

# DEFAULT CONFIG

# Weights
dir_weights_default = 'weights'
weights_meso4_df_default = 'Meso4_DF'
weights_meso4_f2f_default = 'Meso4_F2F'
weights_mesoception_df_default = 'MesoInception_DF'
weights_mesoception_f2f_default = 'MesoInception_F2F'

step_save_weights_temp_default = 5
dir_weights_temp_default = 'weights_temp'

# Training
learning_rate_default = 0.00001
dl_rate_default = 1

number_epochs_default = 10
batch_size_test_default = 10
batch_size_training_default = 10
batch_size_validation_default = 10

target_size_default = (256, 256)

rescale_default = 1 / 255

# dataset
dir_dataset_default = 'dataset'
dir_deepfake_default = 'df'
dir_real_default = 'real'
dir_f2f_default = 'f2f'

def print_info():
    print('\n---------- MesoNet -----------')
    print('Darius Afchar     (ENPC)')
    print('Vincent Nozick    (UPEM-LIGM)')
    print('Junichi Yamagishi (NII)')
    print('Isao Echizen      (NII)')
    print('------------------------------\n')

print_info()

class ClassifierLoader:
    meso4_df          = 'MESO4_DF'
    meso4_f2f          = 'MESO4_F2F'
    mesoception_df = 'MESOCEPTION_DF'
    mesoception_f2f = 'MESOCEPTION_F2F'

    @staticmethod
    def get_classifier(method_classifier,
                       dir_weights          =dir_weights_default,
                       dir_weights_temp     =dir_weights_temp_default,
                       learning_rate        =learning_rate_default,
                       name_weights         = None
                       ):
        switch = {
            ClassifierLoader.meso4_df        : (clf.Meso4, weights_meso4_df_default),
            ClassifierLoader.meso4_f2f       : (clf.Meso4, weights_meso4_f2f_default),
            ClassifierLoader.mesoception_df  : (clf.MesoInception4, weights_mesoception_df_default),
            ClassifierLoader.mesoception_f2f : (clf.MesoInception4, weights_mesoception_df_default)
        }
        pair_classifier = switch.get(method_classifier, None)
        if pair_classifier is None:
            raise ValueError('No classifier called ' + method_classifier)
        functor_classifier, name_weights_default = pair_classifier
        path_dir = os.path.dirname(os.path.realpath(__file__))
        if name_weights is None:
            name_weights = name_weights_default
        path_dir_weights = os.path.join(path_dir, dir_weights)
        path_dir_weights_temp = os.path.join(path_dir, dir_weights_temp)
        classifier = functor_classifier(learning_rate, name_weights, path_dir_weights, path_dir_weights_temp)
        return classifier


def learn_from_dir(name_classifier,
                   dir_dataset,
                   batch_size_training=batch_size_training_default,
                   batch_size_validation=batch_size_validation_default,
                   number_epochs=number_epochs_default,
                   learning_rate=learning_rate_default,
                   step_save_weights_temp=step_save_weights_temp_default,
                   target_size=target_size_default,
                   rescale=rescale_default
                   ):
    data_generator_training, data_generator_validation = lrn.load_data_generators_learning(rescale)
    generator_training,  generator_validation = lrn.load_dataset_learning(dir_dataset,
                                                                          data_generator_training,
                                                                          data_generator_validation,
                                                                          batch_size_training,
                                                                          batch_size_validation,
                                                                          target_size)
    classifier = ClassifierLoader.get_classifier(name_classifier,
                                                 learning_rate=learning_rate,
                                                 name_weights=None)
    lrn.learn_from_generator(classifier,
                             generator_training,
                             generator_validation,
                             batch_size_training,
                             number_epochs,
                             step_save_weights_temp)
    return


def test_from_dir(name_classifier,
                  dir_input,
                  batch_size=batch_size_test_default,
                  learning_rate=learning_rate_default,
                  target_size=target_size_default,
                  rescale=rescale_default
                  ):
    # Flow images from directory and predict
    data_generator_test = tst.load_data_generator_test(rescale=rescale)
    generator_test = tst.load_dataset_test(dir_input,
                                           data_generator_test,
                                           batch_size,
                                           target_size)
    classifier = ClassifierLoader.get_classifier(name_classifier,
                                                 learning_rate=learning_rate,
                                                 name_weights=None)
    tst.test_from_generator(classifier,
                            generator_test,
                            batch_size)
    return
