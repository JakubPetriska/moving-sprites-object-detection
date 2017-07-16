import datetime
import os

from keras.callbacks import TensorBoard, ModelCheckpoint

from detection import constants
from detection.dataset import read_dataset
from detection.model import Model
from detection.utils import generate_video_sequence
from detection.utils import save_masks

LIGHT_OUTPUT = False  # Turns off all following
SAVE_GROUND_TRUTH_TEST_MASKS = True
SAVE_PREDICTED_TEST_MASKS = True
GENERATE_ANNOTATED_VIDEO = True

PROGRESS_VERBOSITY = 1
PLOT_MODEL = True

ALLOWED_OBJECT_TYPES = [
    'car-01.png', 'car-02.png', 'car-03.png', 'car-04.png',
    'car-05.png', 'car-06.png', 'car-07.png', 'car-08.png',
    'car-09.png', 'car-10.png', 'car-11.png', 'car-12.png'
]

# Create model instance
model = Model(verbosity=PROGRESS_VERBOSITY)

model_output_shape = model.model.layers[-1].output_shape[1:]

# Read the training/validation/test data
x_train, y_train = read_dataset(constants.DATASET_TRAINING_PATH,
                                model_output_shape, ALLOWED_OBJECT_TYPES)
x_validation, y_validation = read_dataset(constants.DATASET_TRAINING_PATH,
                                          model_output_shape, ALLOWED_OBJECT_TYPES)
x_test, y_test = read_dataset(constants.DATASET_TEST_PATH,
                              model_output_shape, ALLOWED_OBJECT_TYPES)

# Create output dir if it does not exist
output_dir_name = constants.RESULT_DIR_FORMAT % datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print('\nOutput dir: %s' % output_dir_name)
output_dir = os.path.join(constants.RESULTS_DIR, '%s') % output_dir_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot the model
if PLOT_MODEL:
    from keras.utils.visualize_util import plot

    plot(model.model, to_file=os.path.join(output_dir, constants.RESULT_MODEL_PLOT), show_shapes=True)

# Train the network
print('\nTraining')
tensorboard_callback = TensorBoard(log_dir=output_dir)
model_checkpoint_callback = ModelCheckpoint(os.path.join(output_dir, constants.RESULT_MODEL_BEST_WEIGHTS_FILE),
                                            monitor='val_acc', verbose=PROGRESS_VERBOSITY, save_best_only=True,
                                            save_weights_only=True)
model.train(x_train, y_train, validation_data=(x_validation, y_validation),
            callbacks=[tensorboard_callback, model_checkpoint_callback])

# Save model
print('\nSaving model to disk')
model.save_to_disk(os.path.join(output_dir, constants.RESULT_MODEL_FILE),
                   os.path.join(output_dir, constants.RESULT_MODEL_WEIGHTS_FILE))

# Generate annotated test video sequence
if not LIGHT_OUTPUT:
    print('Creating annotated test data')
    if SAVE_GROUND_TRUTH_TEST_MASKS:
        save_masks(os.path.join(output_dir, constants.RESULT_MASKS_DIR), y_test)
    if SAVE_PREDICTED_TEST_MASKS or GENERATE_ANNOTATED_VIDEO:
        y_predicted = model.predict(x_test)
        if SAVE_PREDICTED_TEST_MASKS:
            save_masks(os.path.join(output_dir, constants.RESULT_PREDICTED_MASKS_DIR), y_predicted)
        if GENERATE_ANNOTATED_VIDEO:
            generate_video_sequence(os.path.join(output_dir, constants.RESULT_VIDEO_FILE),
                                    os.path.join(output_dir, constants.RESULT_IMAGES_DIR),
                                    x_test, y_predicted)
