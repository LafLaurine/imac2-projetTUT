import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def load_data_generator_test(rescale):
    generator_test = ImageDataGenerator(rescale=rescale)
    return generator_test


def load_dataset_test(dir_database,
                      data_generator_test,
                      batch_size_test,
                      target_size: tuple
                     ):
    print(dir_database)
    generator_test = data_generator_test.flow_from_directory(
        dir_database,
        shuffle=False,
        target_size=target_size,
        batch_size=batch_size_test,
        class_mode=None
    )
    print("Flowing images from", dir_database)
    return generator_test

def test_from_generator(classifier,
                        generator_test,
                        batch_size_test,
                        ):

    # 3 - Predict images by batch
    number_images = generator_test.classes.shape[0]
    number_epochs = number_images // batch_size_test
    predicted = np.ones((number_images, 1)) / 2.

    for e in range(number_epochs+1):
        image = generator_test.next()
        print("Epoch ", e + 1, "/", number_epochs + 1)
        prediction = classifier.predict(image)
        predicted[(e * batch_size_test):(e * batch_size_test + len(prediction))] = prediction

    print('Mean prediction  :', np.mean(predicted, axis=0)[0])
    print('Deepfake percent :', np.mean(predicted < 0.5))

    for i in range(len(predicted)):
        image_id = "%3d" % (i)
        print(image_id, ' ', predicted[i])