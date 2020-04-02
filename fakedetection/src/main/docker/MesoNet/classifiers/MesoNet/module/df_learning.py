import numpy as np

from tqdm import tqdm

from .common_classifier import ImageDataGeneratorMeso, GeneratorIterationHandler
from ...common_prediction import EvaluationLearning


def load_data_generators_learning(
        rescale
        # TODO: add as arguments
):
    data_generator_training = ImageDataGeneratorMeso(
        rescale=rescale,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        zoom_range=0.2,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        channel_shift_range=30,
        horizontal_flip=True,
        validation_split=0.1)

    data_generator_validation = ImageDataGeneratorMeso(
        rescale=rescale,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        horizontal_flip=True,
        validation_split=0.1)
    return data_generator_training, data_generator_validation


def load_dataset_learning(
        classifier,
        dir_dataset,
        data_generator_training: ImageDataGeneratorMeso,
        data_generator_validation: ImageDataGeneratorMeso,
        batch_size,
        target_size: tuple,
):
    ## Load dataset
    generator_training = data_generator_training.flow_from_directory(
        classifier=classifier,
        directory=dir_dataset,
        shuffle=True,
        target_size=target_size,
        batch_size=batch_size,  # 75,
        class_mode='binary',
        subset='training')

    generator_validation = data_generator_validation.flow_from_directory(
        classifier=classifier,
        directory=dir_dataset,
        target_size=target_size,
        shuffle=True,
        batch_size=batch_size, # 200
        class_mode='binary',
        subset='validation')

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
    # To handle an odd case of supposedly corrupted PNG file
    it_generator_training = GeneratorIterationHandler(generator_training)
    it_generator_validation = GeneratorIterationHandler(generator_validation)
    evals_learning = EvaluationLearning()
    try:
        for epoch in tqdm(range(number_epochs)):
            mean_loss = 0
            mean_accuracy = 0
            # for image, label in generator_training:
            batch_images_training, batch_labels_training = next(it_generator_training)
            loss = classifier.fit(batch_images_training, batch_labels_training)
            mean_loss += loss[0]
            mean_accuracy += loss[1]

            batch_labels_predicted = classifier.predict(batch_images_training)

            if epoch % step_save_weights_temp == 0:
                # saving weights as a fallback
                classifier.save_weights_temp(epoch)

            loss_training, accuracy_training = np.round(classifier.get_accuracy(batch_images_training, batch_labels_training), decimals=number_decimals)
            #####

            batch_images_validation, batch_labels_validation = next(it_generator_validation)
            loss_validation, accuracy_validation = np.round(classifier.get_accuracy(batch_images_validation, batch_labels_validation), decimals=number_decimals)
            evals_learning.add_eval(epoch=epoch,
                                    loss_training=loss_training,
                                    acc_training=accuracy_training,
                                    loss_validation=loss_validation,
                                    acc_validation=accuracy_validation)
    except KeyboardInterrupt:
        pass
    # saving weights one last time
    classifier.save_weights_temp(epoch)
    return evals_learning

