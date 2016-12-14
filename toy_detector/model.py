from keras.callbacks import History
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adagrad

from common.model_wrapper import ExtendedModelWrapper
from toy_dataset_generator import constants

BATCH_SIZE = 50
TRAINING_EPOCHS = 12


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

    def train(self, x, y, validation_data=None):
        history = History()
        self.model.fit(x, y, nb_epoch=TRAINING_EPOCHS, batch_size=BATCH_SIZE, verbose=self.verbosity,
                       validation_data=validation_data, callbacks=[history])
        return history

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, batch_size=BATCH_SIZE, verbose=self.verbosity)

    def predict(self, x):
        return self.model.predict(x, batch_size=BATCH_SIZE, verbose=self.verbosity)
