from abc import ABC, abstractmethod

from keras.models import model_from_json


class ModelWrapper(ABC):
    """
    Wrapper class for model to have all model related stuff at one place.
    """
    def __init__(self, model_file_path=None, weights_file_path=None, params=None, verbosity=1):
        self.params = params
        if model_file_path:
            with open(model_file_path, 'r') as model_file:
                self.model = model_from_json(model_file.read())
        else:
            self.model = self.build_model()
        self.compile_model()
        if weights_file_path:
            self.model.load_weights(weights_file_path)
        self.verbosity = verbosity

    def save_to_disk(self, model_file_path, weights_file_path):
        with open(model_file_path, 'w') as model_file:
            model_file.write(self.model.to_json())
        self.model.save_weights(weights_file_path)

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def compile_model(self):
        pass


class ExtendedModelWrapper(ModelWrapper):
    """
    Adds more methods to the model wrapper.
    """
    @abstractmethod
    def train(self, x, y, validation_data=None, callbacks=None):
        pass

    @abstractmethod
    def evaluate(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass
