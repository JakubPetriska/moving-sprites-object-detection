import numpy as np
import tensorflow as tf
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adagrad
from tabulate import tabulate

from common.model_wrapper import ExtendedModelWrapper
from object_detector import utils
from object_detector.training_utils import train_and_evaluate
from object_detector.utils import read_kitti_dataset

RESULTS_DIR = 'results_kitti'

PROGRESS_VERBOSITY = 1
PLOT_MODEL = False
DEBUG = False

BATCH_SIZE = 50
TRAINING_EPOCHS = 12

DATASET_TEST_DATA_PERCENTAGE = 0.1

MASK_OBJECTS_IMPORTANCE_MULTIPLIER = 5

NUM_RUNS = 1
ALLOWED_OBJECT_TYPES = ['Car', 'Van', 'Truck']


def loss_function(y_true, y_pred):
    error_scale = tf.scalar_mul(MASK_OBJECTS_IMPORTANCE_MULTIPLIER, y_true)
    error_scale = tf.add(error_scale, tf.ones(tf.shape(error_scale)))
    mse = tf.subtract(y_pred, y_true)
    mse = tf.square(mse)
    mse_scaled = tf.mul(mse, error_scale)
    return tf.scalar_mul(1 / (utils.KITTI_USED_RESOLUTION_HEIGHT * utils.KITTI_ORIGINAL_RESOLUTION_WIDTH),
                         tf.reduce_sum(mse_scaled))


class KittiModel(ExtendedModelWrapper):
    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 7, 7, activation='relu',
                                input_shape=(utils.KITTI_USED_RESOLUTION_HEIGHT, utils.KITTI_USED_RESOLUTION_WIDTH, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(64, 5, 5, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(Convolution2D(32, 1, 1, activation='relu'))
        model.add(Convolution2D(16, 1, 1, activation='relu'))
        model.add(Convolution2D(1, 1, 1, activation='relu'))
        return model

    def compile_model(self):
        self.model.compile(loss=loss_function, optimizer=Adagrad(lr=0.001), metrics=['accuracy'])

    def train(self, x, y, validation_data=None, callbacks=None):
        self.model.fit(x, y, nb_epoch=TRAINING_EPOCHS, batch_size=BATCH_SIZE, verbose=self.verbosity,
                       validation_data=validation_data, callbacks=callbacks)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, batch_size=BATCH_SIZE, verbose=self.verbosity)

    def predict(self, x):
        return self.model.predict(x, batch_size=BATCH_SIZE, verbose=self.verbosity)


for i in range(NUM_RUNS):
    model_wrapper = KittiModel(verbosity=PROGRESS_VERBOSITY)

    model_output_shape = model_wrapper.model.layers[-1].output_shape[1:]

    if not DEBUG:
        x, y = read_kitti_dataset(model_output_shape, allowed_types=ALLOWED_OBJECT_TYPES)
        indices = np.random.permutation(x.shape[0])
        test_samples_count = round(len(indices) * DATASET_TEST_DATA_PERCENTAGE)
        test_indices, training_indices = indices[:test_samples_count], indices[test_samples_count:]
        x_test, y_test = x[test_indices], y[test_indices]
        x_validation, y_validation = x_test, y_test
        x_train, y_train = x[training_indices], y[training_indices]
        print('Dataset size')
        print(tabulate([['Training', x_train.shape[0]], ['Testing', x_test.shape[0]]], headers=['Data', 'Frame count']))
    else:
        x_train, y_train = read_kitti_dataset(model_output_shape, max_frames=100, allowed_types=ALLOWED_OBJECT_TYPES)
        x_test, y_test = x_train, y_train
        x_validation, y_validation = x_test, y_test

    train_and_evaluate(model_wrapper, x_train, y_train, x_validation, y_validation, x_test, y_test,
                       verbosity=PROGRESS_VERBOSITY, plot_model=PLOT_MODEL,
                       results_dir=RESULTS_DIR)
