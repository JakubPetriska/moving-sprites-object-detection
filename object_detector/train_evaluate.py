import datetime
import os

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils.visualize_util import plot
from tabulate import tabulate

from common.utils import get_duration_minutes
from common.utils import start_timer
from object_detector import utils
from object_detector.model import BATCH_SIZE, ToyModel
from object_detector.utils import generate_video_sequence
from object_detector.utils import read_toy_dataset
from object_detector.utils import save_masks
from toy_dataset_generator import constants

RESULT_DIR_FORMAT = os.path.join(os.pardir, os.pardir, 'results', 'result_%s')
TENSORBOARD_LOGS_DIR = 'tensorboard_logs'
MODEL_PLOT = 'model.png'
MASKS_DIR = 'masks_ground_truth'
OUTPUT_INFO_FILE = 'output'

SAVE_GROUND_TRUTH_TEST_MASKS = True
GENERATE_ANNOTATED_VIDEO = True
SAVE_PREDICTED_TEST_MASKS = True
SAVE_LOG_FILE = False

PROGRESS_VERBOSITY = 1
DEBUG = False

result_dir = RESULT_DIR_FORMAT % datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

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
    x_test, y_test = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.TEST_DATASET_PATH), output_shape)
else:
    x_train, y_train = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.VALIDATION_DATASET_PATH),
                                        output_shape)
    x_train = x_train[:2 * BATCH_SIZE]
    y_train = y_train[:2 * BATCH_SIZE]
    x_validation = x_train
    y_validation = y_train
    x_test = x_train
    y_test = y_train

print('Data read in %.2f minutes' % get_duration_minutes(start))

# Do initial evaluation of validation and test data
initial_training_eval = model_wrapper.evaluate(x_train, y_train)
initial_validation_eval = model_wrapper.evaluate(x_validation, y_validation)
initial_test_eval = model_wrapper.evaluate(x_test, y_test)

# Train the network
tensorboard_callback = TensorBoard(log_dir=os.path.join(result_dir, TENSORBOARD_LOGS_DIR))
model_checkpoint_callback = ModelCheckpoint(os.path.join(result_dir, utils.MODEL_BEST_WEIGHTS_FILE),
                                            monitor='val_acc', verbose=PROGRESS_VERBOSITY, save_best_only=True,
                                            save_weights_only=True)
model_wrapper.train(x_train, y_train, validation_data=(x_validation, y_validation),
                    callbacks=[tensorboard_callback, model_checkpoint_callback])

# Save model
print('Saving model to disk')
model_wrapper.save_to_disk(os.path.join(result_dir, utils.MODEL_FILE),
                           os.path.join(result_dir, utils.MODEL_WEIGHTS_FILE))

# Evaluate performance
print("Training finished")
final_training_eval = model_wrapper.evaluate(x_train, y_train)
final_validation_eval = model_wrapper.evaluate(x_validation, y_validation)
final_test_eval = model_wrapper.evaluate(x_test, y_test)

# Generate annotated test video sequence
print('Creating annotated test data')
if SAVE_GROUND_TRUTH_TEST_MASKS:
    save_masks(os.path.join(result_dir, MASKS_DIR), y_test)

if SAVE_PREDICTED_TEST_MASKS or GENERATE_ANNOTATED_VIDEO:
    y_predicted = model_wrapper.predict(x_test)
    if SAVE_PREDICTED_TEST_MASKS:
        save_masks(os.path.join(result_dir, utils.PREDICTED_MASKS_DIR), y_predicted)
    if GENERATE_ANNOTATED_VIDEO:
        generate_video_sequence(os.path.join(result_dir, utils.VIDEO_FILE),
                                os.path.join(result_dir, utils.IMAGES_DIR),
                                x_test, y_predicted)

# Plot the model
plot(model_wrapper.model, to_file=os.path.join(result_dir, MODEL_PLOT), show_shapes=True)

evaluation_table = tabulate([['Training', initial_training_eval[0], initial_training_eval[1],
                              final_training_eval[0], final_training_eval[1]],
                             ['Validation', initial_validation_eval[0], initial_validation_eval[1],
                              final_validation_eval[0], final_validation_eval[1]],
                             ['Test', initial_test_eval[0], initial_test_eval[1],
                              final_test_eval[0], final_test_eval[1]]],
                            headers=['Data', 'Initial loss', 'Initial accuracy', 'Final loss', 'Final accuracy'])
with open(os.path.join(result_dir, OUTPUT_INFO_FILE), mode='w') as output_file:
    output_file.write(evaluation_table)
print('\n' + evaluation_table)
