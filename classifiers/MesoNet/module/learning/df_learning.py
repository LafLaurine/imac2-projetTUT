# -*- coding:utf-8 -*-
# cd Documents/NII/blurDetection

import random
import numpy as np
import matplotlib.pyplot as plt

#from forgery_detection.classifiers import Georges, Hector, Jean
from ..common_classifier import Classifier, Meso4

from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import gaussian_filter


import warnings
warnings.filterwarnings('ignore', '.*Clipping input data to the valid range for imshow*')



## Preprocessing
print('preprocessing ...')

def hide_eyes(batch):
    batch[:, 50:100, 64:192] = 0
    return batch

def hide_face(batch):
    batch[:, 64:192, 64:192] = 0
    return batch

def isolate_face(batch):
    batch[:, :64, :] = 0
    batch[:, 192:, :] = 0
    batch[:, 64:192, :64] = 0
    batch[:, 64:192, 192:] = 0
    return batch

def suppress_context(batch, sigma = 2):
    '''
    Depuis des images dans [0,1]
    '''
    blurred_batch = gaussian_filter(batch, (0, sigma, sigma, 0))
    gaussian_batch = np.abs(batch - blurred_batch)
    processes_batch = gaussian_batch - 0.016
    processes_batch = processes_batch / 0.026
    return processes_batch




def load_data_generators_learning(
        rescale
        # TODO: add as arguments
):
    data_generator_training = ImageDataGenerator(
        rescale=rescale,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        zoom_range=0.2,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        channel_shift_range=30,
        horizontal_flip=True,
        validation_split=0.1)

    data_generator_validation = ImageDataGenerator(
        rescale=rescale,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        horizontal_flip=True,
        validation_split=0.1)
    return data_generator_training, data_generator_validation


def load_dataset_learning(
        dir_dataset,
        data_generator_training: ImageDataGenerator,
        data_generator_validation: ImageDataGenerator,
        batch_size_training,
        batch_size_validation,
        target_size: tuple,
):
    ## Load dataset
    print('\nload dataset ...')
    generator_training = data_generator_training.flow_from_directory(
        dir_dataset,
        target_size=target_size,
        batch_size=batch_size_training,  # 75,
        class_mode='binary',
        subset='training')

    generator_validation = data_generator_validation.flow_from_directory(
        dir_dataset,
        target_size=target_size,
        batch_size=batch_size_validation, # 200
        class_mode='binary',
        subset='validation')

    ## ??
    a, b = generator_training.next()
    return generator_training, generator_validation

def split_classifier(
        classifier,
        split_layer=0
):
    # split_layer : 0, 4 or 6
    # BELOW split_layer  -> frozen
    # ABOVE              -> trainable
    print('\nInitialising classifier ...')
    model = classifier.get_model()
    # TODO: What's that for?
    model.summary()
    ## Freeze layers
    print('\nFreezing', split_layer,  'layers ...')
    for layer in model.layers[:split_layer]:
        layer.trainable = False

    for layer in model.layers[split_layer:]:
        layer.trainable = True
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    # compiling again
    classifier.compile()
    return classifier

def learn_from_generator(
        classifier,
        generator_training,
        generator_validation,
        batch_size,
        number_epochs,
        step_save_weights_temp: int, # Step in epochs at which the weights are saved temporarily
        number_decimals=3
):
    ## Train
    print('\nstart training ...')

    #number_epochs = 101 # 51
    # batch_size = 5
    # residual = False # Not used ? Input images are pre-processed

    # accuracy_plot = []
    # accuracy_validation_plot = []

    try:
        for e in range(number_epochs):
            print('\nepoch', e+1, '/', number_epochs)
        
            batches = 0
            mean_loss = 0
            mean_accuracy = 0
        
            for image, label in generator_training:
                #if residual:
                #    x_batch = suppressContext(x_batch)
                loss = classifier.fit(image, label)
                mean_loss += loss[0]
                mean_accuracy += loss[1]
                batches += 1
                if batches >= batch_size:
                    print('eq', np.round(np.mean(classifier.predict(image)), decimals=number_decimals))
                    break

            if (e % step_save_weights_temp == 0):
                # saving weights as a fallback
                classifier.save_weights_temp(e)
        
            loss_training = np.round(mean_loss / batch_size, decimals=number_decimals)
            accuracy_training = np.round(mean_accuracy / batch_size, decimals=number_decimals)
            validation_a, validation_b = generator_validation.next()
            #if residual:
            #    validation_a = suppressContext(validation_a)
            loss_validation, accuracy_validation = np.round(classifier.get_accuracy(validation_a, validation_b), decimals=number_decimals)
            print("LossTr", loss_training, "%Tr", accuracy_training, "LossVal", loss_validation, "%Val", accuracy_validation)
        
       # accuracy_training_plot.append(accuracy_training)
       # accuracy_validation_plot.append(accuracy_validation)

    except KeyboardInterrupt:
        pass
    ## Test accuracy
    print('\ntest accuracy ...')

    validation_a, validation_b = generator_validation.next()
    # validation_a = suppressContext(validation_a)
    print(validation_a.shape)
    print('Complete (?)', classifier.get_accuracy(validation_a, validation_b))

    # validation_c = isolateFace(np.copy(validation_a))
    # print('Visage uniquement', model.get_accuracy(validation_c, validation_b))
    #
    # validation_d = hideFace(np.copy(validation_a))
    # print('Fond', model.get_accuracy(validation_d, validation_b))

    ## Smoothe curve ?

def smoothe_curve(y, window = 5):
    n = len(y)
    yp = np.array(y)
    zp = np.zeros(n)
    for i in range(window, n-window):
        zp[i] = np.mean(yp[(i-window):(i+window+1)])
    return zp


def plot_curves(accuracy_training_plot, accuracy_validation_plot):
    ## Plot learning curves
    print('\nplot curves ...')
    plt.figure()
    plt.plot(accuracy_training_plot, 'b')
    plt.plot(accuracy_validation_plot, 'r-')
    plt.show()

## ## ----- OUTPUT

def display_sample(generator):
    ## Display sample
    print('\ndisplay sample ...')
    a, b = generator.next()
    plt.figure()
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.imshow(a[i, :, :])
        plt.axis('off')
    plt.show()



def find_worst_classified(classifier, generator_validation):
    ## Find worst classified
    print('\nfind worst classified ...')
    validation_a, validation_b = generator_validation.next()
    indices_bad = []
    validation_z = classifier.predict(validation_a)
    validation_diff = np.abs(validation_b.flatten() - validation_z.flatten())
    for i in range(validation_b.shape[0]):
        if validation_diff[i] > 0.5:
            indices_bad.append(i)
    print(len(indices_bad), "found")

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(validation_a[indices_bad[i], :, :, 0], cmap="Greys_r")
        plt.axis('off')
    plt.show()

## ----- HIDDEN LAYERS



def plot_weights(classifier):
    ## Get weights
    print('\nplot weights ...')

    weights = classifier.get_weights()
    weights_conv = weights[0]
    plt.figure()
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(weights_conv[:, :, :, i], interpolation="none")
        plt.axis('off')
    plt.show()


def show_intermidiate(classifier, generator, accuracy_plot):
    ## Show intermediate representation
    a, b = generator.next()
    print('\nshow layers ...')
    model = classifier.get_model()
    img = a[0:1]
    l1 = model.x1.predict(img)
    l2 = model.x2.predict(img)
    l3 = model.x3.predict(img)
    plt.figure()
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.axis('off')
        plt.imshow(l1[0, :, :, i])

    plt.figure()
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.axis('off')
        plt.imshow(l2[0, :, :, i])

    plt.figure()
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.axis('off')
        plt.imshow(l3[0, :, :, i])

    plt.show()
    ##
    for i in accuracy_plot[80:]:
        print(i)