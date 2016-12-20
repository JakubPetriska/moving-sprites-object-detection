import datetime
import os
import sys

import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.utils.visualize_util import plot

from common.loggers import LoggerErr
from common.loggers import LoggerOut
from common.utils import get_duration_minutes
from common.utils import start_timer
from toy_dataset_generator import constants
from toy_detector.model import BATCH_SIZE, ToyModel
from toy_detector.utils import generate_video_sequence
from toy_detector.utils import read_toy_dataset
from toy_detector.utils import save_masks

RESULT_DIR_FORMAT = os.path.join(os.pardir, os.pardir, 'results', 'result_%s')
TENSORBOARD_LOGS_DIR = 'tensorboard_logs'
MODEL_PLOT = 'model.png'
IMAGES_DIR = 'images_annotated'
MASKS_DIR = 'masks_ground_truth'
PREDICTED_MASKS_DIR = 'masks_predicted'
VIDEO_FILE = 'video.mp4'
VALIDATION_ERROR_GRAPH_FILE = 'validation_accuracy.png'
CONSOLE_OUTPUT_FILE = 'output.txt'
MODEL_FILE = 'model.json'
MODEL_WEIGHTS_FILE = 'model_weights.h5'

SAVE_GROUND_TRUTH_TEST_MASKS = True
GENERATE_ANNOTATED_VIDEO = True
SAVE_PREDICTED_TEST_MASKS = True
SAVE_LOG_FILE = False

PROGRESS_VERBOSITY = 1
DEBUG = False

result_dir = RESULT_DIR_FORMAT % datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if SAVE_LOG_FILE:
    output_file_path = os.path.join(result_dir, CONSOLE_OUTPUT_FILE)
    sys.stdout = LoggerOut(output_file_path)
    sys.stderr = LoggerErr(output_file_path)

# Create the model
model_wrapper = ToyModel(verbosity=PROGRESS_VERBOSITY)

output_shape = model_wrapper.model.layers[-1].output_shape[1:]

# Read data
print('Reading training and validation data')
start = start_timer()
if not DEBUG:
    x_train, y_train = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.TRAINING_DATASET_PATH),
                                        output_shape)
    x_validation, y_validation = read_toy_dataset(
        os.path.join(constants.OUTPUT_PATH, constants.VALIDATION_DATASET_PATH),
        output_shape)
else:
    x_train, y_train = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.VALIDATION_DATASET_PATH),
                                        output_shape)
    x_train = x_train[:2 * BATCH_SIZE]
    y_train = y_train[:2 * BATCH_SIZE]
    x_validation = x_train
    y_validation = y_train

print('Data read in %.2f minutes' % get_duration_minutes(start))

# Train the network
# training_history = model_wrapper.train(x_train, y_train, validation_data=(x_validation, y_validation))
tensorboard_callback = TensorBoard(log_dir=os.path.join(result_dir, TENSORBOARD_LOGS_DIR))
model_wrapper.train(x_train, y_train, validation_data=(x_validation, y_validation), callbacks=[tensorboard_callback])

# Save model
print('Saving model to disk')
model_wrapper.save_to_disk(os.path.join(result_dir, MODEL_FILE), os.path.join(result_dir, MODEL_WEIGHTS_FILE))

# Evaluate performance
print("Training finished")
x_test, y_test = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.TEST_DATASET_PATH), output_shape)
if DEBUG:
    x_test = x_test[:2 * BATCH_SIZE]
    y_test = y_test[:2 * BATCH_SIZE]

test_error = model_wrapper.evaluate(x_test, y_test)
print('\tTest result - loss: %s, accuracy: %s' % tuple(test_error))

# Generate annotated test video sequence
print('Creating annotated test data')
if SAVE_GROUND_TRUTH_TEST_MASKS:
    save_masks(os.path.join(result_dir, MASKS_DIR), y_test)

if SAVE_PREDICTED_TEST_MASKS or GENERATE_ANNOTATED_VIDEO:
    y_predicted = model_wrapper.predict(x_test)
    if SAVE_PREDICTED_TEST_MASKS:
        save_masks(os.path.join(result_dir, PREDICTED_MASKS_DIR), y_predicted)
    if GENERATE_ANNOTATED_VIDEO:
        generate_video_sequence(os.path.join(result_dir, VIDEO_FILE), os.path.join(result_dir, IMAGES_DIR),
                                x_test, y_predicted)

# Plot the model
plot(model_wrapper.model, to_file=os.path.join(result_dir, MODEL_PLOT), show_shapes=True)
