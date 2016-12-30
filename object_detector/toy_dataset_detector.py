import os

import sys
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adagrad

from common.model_wrapper import ExtendedModelWrapper
from object_detector.utils import read_toy_dataset
from toy_dataset_generator import constants
from object_detector.training_utils import train_and_evaluate

RESULTS_DIR = 'results_toy_dataset'

PROGRESS_VERBOSITY = 1
PLOT_MODEL = False
DEBUG = False

BATCH_SIZE = 50
TRAINING_EPOCHS = 12

NUM_RUNS = 3
ALLOWED_OBJECT_TYPES = ['1', '2']


class ToyModel(ExtendedModelWrapper):
    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 7, 7, activation='relu',
                                input_shape=(constants.RESOLUTION_HEIGHT, constants.RESOLUTION_WIDTH, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(64, 5, 5, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(Convolution2D(32, 1, 1, activation='relu'))
        model.add(Convolution2D(16, 1, 1, activation='relu'))
        model.add(Convolution2D(1, 1, 1, activation='relu'))
        return model

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer=Adagrad(lr=0.001), metrics=['accuracy'])

    def train(self, x, y, validation_data=None, callbacks=None):
        self.model.fit(x, y, nb_epoch=TRAINING_EPOCHS, batch_size=BATCH_SIZE, verbose=self.verbosity,
                       validation_data=validation_data, callbacks=callbacks)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, batch_size=BATCH_SIZE, verbose=self.verbosity)

    def predict(self, x):
        return self.model.predict(x, batch_size=BATCH_SIZE, verbose=self.verbosity)


for i in range(NUM_RUNS):
    model_wrapper = ToyModel(verbosity=PROGRESS_VERBOSITY)

    model_output_shape = model_wrapper.model.layers[-1].output_shape[1:]

    # Change the working directory if the script was executed from somewhere else
    os.chdir(os.path.dirname(sys.argv[0]))
    if not DEBUG:
        x_train, y_train = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.TRAINING_DATASET_PATH),
                                            model_output_shape, allowed_types=ALLOWED_OBJECT_TYPES)
        x_validation, y_validation = read_toy_dataset(
            os.path.join(constants.OUTPUT_PATH, constants.VALIDATION_DATASET_PATH),
            model_output_shape, allowed_types=ALLOWED_OBJECT_TYPES)
        x_test, y_test = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.TEST_DATASET_PATH),
                                          model_output_shape, allowed_types=ALLOWED_OBJECT_TYPES)
    else:
        x_train, y_train = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.TEST_DATASET_PATH),
                                            model_output_shape, allowed_types=ALLOWED_OBJECT_TYPES)
        x_train = x_train[:2 * BATCH_SIZE]
        y_train = y_train[:2 * BATCH_SIZE]
        x_validation = x_train
        y_validation = y_train
        x_test = x_train
        y_test = y_train

    train_and_evaluate(model_wrapper, x_train, y_train, x_validation, y_validation, x_test, y_test,
                       verbosity=PROGRESS_VERBOSITY, plot_model=PLOT_MODEL,
                       results_dir=RESULTS_DIR)
