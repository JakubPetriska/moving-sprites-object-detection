import tensorflow as tf
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adagrad
from detection import constants

from utils.base_model import BaseModel

BATCH_SIZE = 5  # TODO return to 50
TRAINING_EPOCHS = 2  # TODO return to 12

MASK_OBJECTS_IMPORTANCE_MULTIPLIER = 2


def loss_function(y_true, y_pred):
    error_scale = tf.scalar_mul(MASK_OBJECTS_IMPORTANCE_MULTIPLIER, y_true)
    error_scale = tf.add(error_scale, tf.ones(tf.shape(error_scale)))
    mse = tf.subtract(y_pred, y_true)
    mse = tf.square(mse)
    mse_scaled = tf.mul(mse, error_scale)
    return tf.scalar_mul(1 / (constants.RESOLUTION_HEIGHT * constants.RESOLUTION_WIDTH),
                         tf.reduce_sum(mse_scaled))


class Model(BaseModel):
    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 7, 7, activation='relu',
                                input_shape=(constants.RESOLUTION_HEIGHT,
                                             constants.RESOLUTION_WIDTH,
                                             3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(64, 5, 5, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(Convolution2D(32, 1, 1, activation='relu'))
        model.add(Convolution2D(16, 1, 1, activation='relu'))
        model.add(Convolution2D(1, 1, 1, activation='relu'))
        return model

    def compile_model(self):
        self.model.compile(loss=loss_function,
                           optimizer=Adagrad(lr=0.001),
                           metrics=['accuracy'])

    def train(self, x, y, validation_data=None, callbacks=None):
        self.model.fit(x, y,
                       nb_epoch=TRAINING_EPOCHS,
                       batch_size=BATCH_SIZE,
                       verbose=self.verbosity,
                       validation_data=validation_data,
                       callbacks=callbacks)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y,
                                   batch_size=BATCH_SIZE,
                                   verbose=self.verbosity)

    def predict(self, x):
        return self.model.predict(x,
                                  batch_size=BATCH_SIZE,
                                  verbose=self.verbosity)
