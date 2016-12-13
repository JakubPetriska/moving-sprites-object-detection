import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adagrad
from scipy import misc

from common import utils
from common.loggers import LoggerErr
from common.loggers import LoggerOut
from toy_dataset_generator import constants
from toy_dataset_generator import generator_utils

RESULT_DIR_FORMAT = os.path.join('results', 'result_%s')
IMAGES_DIR = 'images_annotated'
MASKS_DIR = 'masks_ground_truth'
PREDICTED_MASKS_DIR = 'masks_predicted'
VIDEO_FILE = 'video.mp4'
VALIDATION_ERROR_GRAPH_FILE = 'validation_error.png'
CONSOLE_OUTPUT_FILE = 'output.txt'
MODEL_FILE = 'model.json'
MODEL_WEIGHTS_FILE = 'model_weights.h5'

SAVE_GROUND_TRUTH_TEST_MASKS = True
GENERATE_ANNOTATED_VIDEO = True
SAVE_PREDICTED_TEST_MASKS = True

PROGRESS_VERBOSITY = 1

VALIDATE_AFTER_EACH_EPOCH = True

TEST_SEQUENCE_OVERLAY_ALPHA = 0.1
TEST_SEQUENCE_OVERLAY_COLOR = [0, 255, 0]

BATCH_SIZE = 50
TRAINING_EPOCHS = 12

DEBUG = False


def read_toy_dataset(path, model_output_shape):
    labels = generator_utils.read_labels(os.path.join(path, constants.DATASET_LABELS_FILE))
    image_path_format = os.path.join(path, constants.DATASET_IMAGES_DIR, constants.FRAME_IMAGE_FILE_NAME_FORMAT)
    x = np.empty((len(labels), constants.RESOLUTION_HEIGHT, constants.RESOLUTION_WIDTH, 3))
    y_shape = [len(labels)]
    y_shape += model_output_shape
    y = np.empty(y_shape)

    vertical_scale_factor = model_output_shape[0] / constants.RESOLUTION_HEIGHT
    horizontal_scale_factor = model_output_shape[0] / constants.RESOLUTION_WIDTH
    for i in range(0, len(labels)):
        label = labels[i]
        image = misc.imread(image_path_format % label[0])
        mask = np.zeros(model_output_shape)
        for object_bounding_box in label[1:]:
            scaled_vertical_bounds = [min(round(bound * vertical_scale_factor), model_output_shape[0] - 1)
                                      for bound in object_bounding_box[:2]]
            scaled_horizontal_bounds = [min(round(bound * horizontal_scale_factor), model_output_shape[1] - 1)
                                        for bound in object_bounding_box[2:]]
            top = scaled_vertical_bounds[0]
            bottom = scaled_vertical_bounds[1]
            left = scaled_horizontal_bounds[0]
            right = scaled_horizontal_bounds[1]
            mask[top:bottom + 1, left:right + 1] = 1
        x[i] = image
        y[i] = mask
    return x, y


def save_masks(path, masks):
    if not os.path.exists(path):
        os.makedirs(path)
    mask_image_path = os.path.join(path, constants.FRAME_IMAGE_FILE_NAME_FORMAT)
    for i in range(len(masks)):
        mask_image = masks[i]
        mask_image *= 255
        mask_image = np.clip(mask_image, 0, 255)
        mask_image = np.round(mask_image).astype(np.uint8)
        mask_image = np.repeat(mask_image, 3, axis=2)
        misc.imsave(mask_image_path % i, mask_image)


def generate_video_sequence(path, images, masks):
    images_dir = os.path.join(path, IMAGES_DIR)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    frame_image_path = os.path.join(images_dir, constants.FRAME_IMAGE_FILE_NAME_FORMAT)
    mask_shape = masks[0].shape
    for i in range(len(images)):
        image = images[i]
        mask = np.clip(masks[i], 0, 1)
        overlay_mask = np.empty(shape=(1, 1, 3))
        overlay_mask[0, 0] = TEST_SEQUENCE_OVERLAY_COLOR
        overlay_mask = np.repeat(overlay_mask, mask_shape[0], axis=0)
        overlay_mask = np.repeat(overlay_mask, mask_shape[1], axis=1)
        overlay_mask *= mask
        overlay_mask *= TEST_SEQUENCE_OVERLAY_ALPHA
        scaled_overlay_mask = misc.imresize(overlay_mask.astype(np.uint8),
                                            (constants.RESOLUTION_HEIGHT, constants.RESOLUTION_WIDTH, 3))
        background_alpha = -mask
        background_alpha += 1
        background_alpha *= 255
        background_alpha = np.repeat(background_alpha, 3, axis=2).astype(np.uint8)
        background_alpha = misc.imresize(background_alpha, (constants.RESOLUTION_HEIGHT, constants.RESOLUTION_WIDTH, 3))
        background_alpha = background_alpha.astype(np.float) / 255
        image *= background_alpha
        image += scaled_overlay_mask
        misc.imsave(frame_image_path % i, image.astype(np.uint8))
    generator_utils.create_video(images_dir, os.path.join(path, VIDEO_FILE))


result_dir = RESULT_DIR_FORMAT % datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

output_file_path = os.path.join(result_dir, CONSOLE_OUTPUT_FILE)
sys.stdout = LoggerOut(output_file_path)
sys.stderr = LoggerErr(output_file_path)

# Create the model
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

model.compile(loss='binary_crossentropy', optimizer=Adagrad(lr=0.001), metrics=['accuracy'])

output_shape = model.layers[-1].output_shape[1:]

# Read data
print('Reading training data')
start = utils.start_timer()

if not DEBUG:
    x_train, y_train = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.TRAINING_DATASET_PATH),
                                        output_shape)
else:
    x_train, y_train = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.VALIDATION_DATASET_PATH),
                                        output_shape)
    x_train = x_train[:2 * BATCH_SIZE]
    y_train = y_train[:2 * BATCH_SIZE]

print('Training data read in %.2f minutes' % utils.get_duration_minutes(start))

# Train the network
validation_results = []
if VALIDATE_AFTER_EACH_EPOCH:
    print('Reading validation data')
    start = utils.start_timer()
    if not DEBUG:
        x_validation, y_validation = read_toy_dataset(
            os.path.join(constants.OUTPUT_PATH, constants.VALIDATION_DATASET_PATH),
            output_shape)
    else:
        x_validation = x_train
        y_validation = y_train
    print('Validation data read in %.2f minutes' % utils.get_duration_minutes(start))
    for i in range(0, TRAINING_EPOCHS):
        print('Training epoch %s' % (i + 1))
        start = utils.start_timer()
        model.fit(x_train, y_train, nb_epoch=1, batch_size=BATCH_SIZE, verbose=PROGRESS_VERBOSITY)
        training_duration = utils.get_duration_minutes(start)
        print('\tEpoch %s trained in %.2f minutes' % (i + 1, training_duration))

        start = utils.start_timer()
        error = model.evaluate(x_validation, y_validation, batch_size=BATCH_SIZE, verbose=PROGRESS_VERBOSITY)
        evaluation_duration = utils.get_duration_minutes(start)
        print('\tModel evaluated in %.2f minutes' % evaluation_duration)
        print('\tValidation result - loss: %s, accuracy: %s' % tuple(error))
        validation_results.append(error)
else:
    model.fit(x_train, y_train, nb_epoch=TRAINING_EPOCHS, batch_size=BATCH_SIZE, verbose=PROGRESS_VERBOSITY)

# Evaluate performance
print("Training finished")
x_test, y_test = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.TEST_DATASET_PATH),
                                  output_shape)
test_error = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=PROGRESS_VERBOSITY)
print('\tTest result - loss: %s, accuracy: %s' % tuple(test_error))

# Save model
print('Saving model to disk')
with open(os.path.join(result_dir, MODEL_FILE), 'w') as model_file:
    model_file.write(model.to_json())
model.save_weights(os.path.join(result_dir, MODEL_WEIGHTS_FILE))

# Generate annotated test video sequence
print('Creating annotated test data')
if SAVE_GROUND_TRUTH_TEST_MASKS:
    save_masks(os.path.join(result_dir, MASKS_DIR), y_test)

if SAVE_PREDICTED_TEST_MASKS or GENERATE_ANNOTATED_VIDEO:
    y_predicted = model.predict(x_test, batch_size=BATCH_SIZE, verbose=PROGRESS_VERBOSITY)
    if SAVE_PREDICTED_TEST_MASKS:
        save_masks(os.path.join(result_dir, PREDICTED_MASKS_DIR), y_predicted)
    if GENERATE_ANNOTATED_VIDEO:
        generate_video_sequence(result_dir, x_test, y_predicted)

# Plot validation error development
if len(validation_results) > 0:
    graph_x = range(1, len(validation_results) + 1)
    validation_results = np.transpose(np.array(validation_results))
    print('Validation performance data: %s' % str(model.metrics_names))
    print(validation_results)
    line = plt.plot(graph_x, validation_results[1])
    plt.title('Prediction accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Prediction accuracy')
    plt.xticks(graph_x)
    plt.grid()
    plt.draw()
    plt.savefig(os.path.join(result_dir, VALIDATION_ERROR_GRAPH_FILE))
