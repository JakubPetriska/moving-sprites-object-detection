import datetime
import os

from keras.callbacks import TensorBoard, ModelCheckpoint
from tabulate import tabulate

from detection import constants
from detection import utils
from detection.dataset import read_dataset
from detection.model import Model
from detection.utils import generate_video_sequence
from detection.utils import save_masks

RESULTS_DIR = 'models'
RESULT_DIR_FORMAT = 'model_%s'
MODEL_PLOT = 'model.png'
MASKS_DIR = 'masks_ground_truth'
OUTPUT_INFO_FILE = 'output'

LIGHT_OUTPUT = False  # Turns off all following
SAVE_GROUND_TRUTH_TEST_MASKS = True
SAVE_PREDICTED_TEST_MASKS = True
GENERATE_ANNOTATED_VIDEO = True

PROGRESS_VERBOSITY = 1
PLOT_MODEL = True

model = Model(verbosity=PROGRESS_VERBOSITY)

model_output_shape = model.model.layers[-1].output_shape[1:]

x_train, y_train = \
    read_dataset(constants.DATASET_TRAINING_PATH, model_output_shape)
x_validation, y_validation = \
    read_dataset(constants.DATASET_TRAINING_PATH, model_output_shape)
x_test, y_test = \
    read_dataset(constants.DATASET_TEST_PATH, model_output_shape)

output_dir_name = RESULT_DIR_FORMAT % datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print('\nOutput dir: %s' % output_dir_name)
output_dir = os.path.join(RESULTS_DIR, '%s') % output_dir_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Do initial evaluation of validation and test data
initial_training_eval = model.evaluate(x_train, y_train)
initial_validation_eval = model.evaluate(x_validation, y_validation)
initial_test_eval = model.evaluate(x_test, y_test)

# Train the network
tensorboard_callback = TensorBoard(log_dir=output_dir)
model_checkpoint_callback = ModelCheckpoint(os.path.join(output_dir, utils.MODEL_BEST_WEIGHTS_FILE),
                                            monitor='val_acc', verbose=PROGRESS_VERBOSITY, save_best_only=True,
                                            save_weights_only=True)
model.train(x_train, y_train, validation_data=(x_validation, y_validation),
            callbacks=[tensorboard_callback, model_checkpoint_callback])

# Save model
print('Saving model to disk')
model.save_to_disk(os.path.join(output_dir, utils.MODEL_FILE),
                   os.path.join(output_dir, utils.MODEL_WEIGHTS_FILE))

# Evaluate performance
print("Training finished")
final_training_eval = model.evaluate(x_train, y_train)
final_validation_eval = model.evaluate(x_validation, y_validation)
final_test_eval = model.evaluate(x_test, y_test)

# Generate annotated test video sequence
if not LIGHT_OUTPUT:
    print('Creating annotated test data')
    if SAVE_GROUND_TRUTH_TEST_MASKS:
        save_masks(os.path.join(output_dir, MASKS_DIR), y_test)
    if SAVE_PREDICTED_TEST_MASKS or GENERATE_ANNOTATED_VIDEO:
        y_predicted = model.predict(x_test)
        if SAVE_PREDICTED_TEST_MASKS:
            save_masks(os.path.join(output_dir, utils.PREDICTED_MASKS_DIR), y_predicted)
        if GENERATE_ANNOTATED_VIDEO:
            generate_video_sequence(os.path.join(output_dir, utils.VIDEO_FILE),
                                    os.path.join(output_dir, utils.IMAGES_DIR),
                                    x_test, y_predicted)

evaluation_table = tabulate([['Training', initial_training_eval[0], initial_training_eval[1],
                              final_training_eval[0], final_training_eval[1]],
                             ['Validation', initial_validation_eval[0], initial_validation_eval[1],
                              final_validation_eval[0], final_validation_eval[1]],
                             ['Test', initial_test_eval[0], initial_test_eval[1],
                              final_test_eval[0], final_test_eval[1]]],
                            headers=['Data', 'Initial loss', 'Initial accuracy', 'Final loss', 'Final accuracy'])
with open(os.path.join(output_dir, OUTPUT_INFO_FILE), mode='w') as output_file:
    output_file.write(evaluation_table)
print('\n' + evaluation_table)

# Plot the model
if PLOT_MODEL:
    from keras.utils.visualize_util import plot

    plot(model.model, to_file=os.path.join(output_dir, MODEL_PLOT), show_shapes=True)
