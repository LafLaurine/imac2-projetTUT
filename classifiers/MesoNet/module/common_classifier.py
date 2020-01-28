# -*- coding:utf-8 -*-

import os

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam

IMGWIDTH = 256

class Classifier:
    # _learning_rate
    # _model : KerasModel
    # _name_weights
    # _path_dir_weights
    # _path_dir_weights_temp
    def __init__(self,
                 model,
                 learning_rate,
                 name_weights,
                 path_dir_weights,
                 path_dir_weights_temp
                 ):
        self._model = model
        self._learning_rate = float(learning_rate)
        self.compile()
        self._name_weights = name_weights
        self._path_dir_weights = path_dir_weights
        self._path_dir_weights_temp = path_dir_weights_temp
        self.load_weights()

    def predict(self, x):
        return self._model.predict(x)

    def fit(self, x, y):
        return self._model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self._model.test_on_batch(x, y)

    def load_weights(self):
        self._model.load_weights(self._path_dir_weights + os.sep + self._name_weights)

    def compile(self):
        optimiser = Adam(lr=self._learning_rate)
        self._model.compile(optimizer=optimiser,
                            loss='mean_squared_error',
                            metrics=['accuracy'])

    def get_model(self):
        return self._model

    def get_weights(self):
        return self._model.get_weights()


    def save_weights(self):
        make_dir_if_not_exist(self._path_dir_weights)
        path_weights = self._path_dir_weights + os.sep + self._name_weights
        self._model.save_weights(path_weights)

    def save_weights_temp(self, epoch):
        make_dir_if_not_exist(self._path_dir_weights_temp)
        path_weights_temp = self._path_dir_weights_temp + os.sep + self._name_weights + '_' + str(epoch)
        self._model.save_weights(path_weights_temp)


class Meso4(Classifier):
    def __init__(self,
                 learning_rate,
                 name_weights,
                 path_dir_weights,
                 path_dir_weights_temp
                 ):
        self._model = self.__init_model()
        super().__init__(self._model, learning_rate, name_weights, path_dir_weights, path_dir_weights_temp)

    def __init_model(self):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return KerasModel(inputs = x, outputs = y)


class MesoInception4(Classifier):
    def __init__(self,
                 learning_rate,
                 name_weights,
                 path_dir_weights,
                 path_dir_weights_temp
                 ):
        self._model = self.__init_model()
        super().__init__(self._model, learning_rate, name_weights, path_dir_weights, path_dir_weights_temp)

    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
            
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
            
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)
            
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])
            
            return y
        return func
    
    def __init_model(self):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return KerasModel(inputs = x, outputs = y)

def make_dir_if_not_exist(path_dir):
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    elif not os.path.isdir(path_dir):
        raise NotADirectoryError(path_dir)
