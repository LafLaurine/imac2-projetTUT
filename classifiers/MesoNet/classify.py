import os
import numpy as np

from .module.learning import df_learning as lrn
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

# database
dir_database_default = 'database'
dir_deepfake_default = 'df'
dir_real_default = 'real'
dir_f2f_default = 'f2f'

database_path_default = dir_database_default + dir_deepfake_default


def print_info():
    print('\n---------- MesoNet -----------')
    print('Darius Afchar     (ENPC)')
    print('Vincent Nozick    (UPEM-LIGM)')
    print('Junichi Yamagishi (NII)')
    print('Isao Echizen      (NII)')
    print('------------------------------\n')


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
            raise ValueError('No classifier named: ' + method_classifier)
        functor_classifier, name_weights_default = pair_classifier
        path_dir = os.path.dirname(os.path.realpath(__file__))
        if name_weights is None:
            name_weights = name_weights_default
        path_dir_weights = path_dir + os.sep + dir_weights
        path_dir_weights_temp = path_dir + os.sep + dir_weights_temp
        classifier = functor_classifier(learning_rate, name_weights, path_dir_weights, path_dir_weights_temp)
        return classifier

def learn_from_dir(name_classifier,
                   path_database,
                   batch_size_training=batch_size_training_default,
                   batch_size_validation=batch_size_validation_default,
                   number_epochs=number_epochs_default,
                   learning_rate=learning_rate_default,
                   step_save_weights_temp=step_save_weights_temp_default,
                   dir_weights_temp=dir_weights_temp_default,
                   target_size=target_size_default,
                   rescale=rescale_default
                   ):
    print_info()
    data_generator_training, data_generator_validation = lrn.load_generators_learning(rescale)
    generator_training,  generator_validation = lrn.load_dataset_learning(path_database,
                                                                            data_generator_training,
                                                                            data_generator_validation,
                                                                            batch_size_training,
                                                                            batch_size_validation,
                                                                            target_size)
    classifier = ClassifierLoader.get_classifier(name_classifier, learning_rate=learning_rate, name_weights=None)
    lrn.learn_from_generator(classifier,
                                  generator_training,
                                  generator_validation,
                                  batch_size_training,
                                  number_epochs,
                                  step_save_weights_temp,
                                  dir_weights_temp)



def test_from_dir(
        classifier_name,
        database_path,
        target_size=target_size_default,
        batch_size=batch_size_test_default,
        rescale=rescale_default
):
    # 1 - Load the model and its pretrained weights
    classifier = ClassifierLoader.get_classifier(classifier_name)

    # 2 - Flow images from directory and predict
    data_generator = ImageDataGenerator(rescale=rescale)


    generator = data_generator.flow_from_directory(
        database_path,
        shuffle=False,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    print("Flowing images from", database_path)

    # 3 - Predict images by batch
    number_images = generator.classes.shape[0]
    number_epochs = n_images // batchSize

    predicted = np.ones((number_images, 1)) / 2.

    for e in range(number_epochs + 1):
        X, Y = generator.next()
        print("Epoch ", e + 1, "/", number_epochs + 1)
        prediction = classifier.predict(X)
        predicted[(e * batchSize):(e * batchSize + prediction.shape[0])] = prediction

    print('Mean prediction  :', np.mean(predicted, axis=0)[0])
    print('Deepfake percent :', np.mean(predicted < 0.5))

    for i in range(len(predicted)):
        image_id = "%3d" % (i)
        print(image_id,' ',predicted[i])