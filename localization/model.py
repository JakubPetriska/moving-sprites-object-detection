import tensorflow as tf
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adagrad

from utils_lib.model_wrapper import ExtendedModelWrapper

PARAM_INPUT_HEIGHT = 'input_height'
PARAM_INPUT_WIDTH = 'input_width'

MASK_OBJECTS_IMPORTANCE_MULTIPLIER = 8

BATCH_SIZE = 50
TRAINING_EPOCHS = 1  # TODO fix


class KittiModel(ExtendedModelWrapper):
    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 7, 7, activation='relu',
                                input_shape=(self.params[PARAM_INPUT_HEIGHT], self.params[PARAM_INPUT_WIDTH], 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(64, 5, 5, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(Convolution2D(32, 1, 1, activation='relu'))
        model.add(Convolution2D(16, 1, 1, activation='relu'))
        model.add(Convolution2D(1, 1, 1, activation='relu'))
        return model

    def loss_function(self, y_true, y_pred):
        error_scale = tf.scalar_mul(MASK_OBJECTS_IMPORTANCE_MULTIPLIER, y_true)
        error_scale = tf.add(error_scale, tf.ones(tf.shape(error_scale)))
        mse = tf.subtract(y_pred, y_true)
        mse = tf.square(mse)
        mse_scaled = tf.mul(mse, error_scale)
        return tf.scalar_mul(1 / (self.params[PARAM_INPUT_HEIGHT] * self.params[PARAM_INPUT_WIDTH]),
                             tf.reduce_sum(mse_scaled))

    def compile_model(self):
        self.model.compile(loss=self.loss_function, optimizer=Adagrad(lr=0.001))

    def train(self, x, y, validation_data=None, callbacks=None):
        if not callbacks:
            callbacks = []
        self.model.fit(x, y, nb_epoch=TRAINING_EPOCHS, batch_size=BATCH_SIZE, verbose=self.verbosity,
                       validation_data=validation_data, callbacks=callbacks)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, batch_size=BATCH_SIZE, verbose=self.verbosity)

    def predict(self, x):
        return self.model.predict(x, batch_size=BATCH_SIZE, verbose=self.verbosity)
