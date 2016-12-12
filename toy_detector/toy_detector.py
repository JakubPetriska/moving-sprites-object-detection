import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from scipy import misc

from common import utils
from toy_dataset_generator import constants
from toy_dataset_generator import generator_utils

RESULT_DIR = 'result'
IMAGES_DIR = 'test_data_images_annotated'
VIDEO_FILE = 'test_data_video.mp4'

TEST_SEQUENCE_OVERLAY_ALPHA = 0.2
TEST_SEQUENCE_OVERLAY_COLOR = [0, 255, 0]

BATCH_SIZE = 50
TRAINING_EPOCHS = 1


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
            scaled_vertical_bounds = [max(round(bound * vertical_scale_factor), model_output_shape[0] - 1)
                                      for bound in object_bounding_box[:2]]
            scaled_horizontal_bounds = [max(round(bound * horizontal_scale_factor), model_output_shape[1] - 1)
                                        for bound in object_bounding_box[2:]]
            top = scaled_vertical_bounds[0]
            bottom = scaled_vertical_bounds[1]
            left = scaled_horizontal_bounds[0]
            right = scaled_horizontal_bounds[1]
            mask[top:bottom + 1, left:right + 1] = 1
        x[i] = image
        y[i] = mask
    return x, y


def generate_video_sequence(path, images, masks):
    images_dir = os.path.join(path, IMAGES_DIR)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    frame_image_path = os.path.join(images_dir, constants.FRAME_IMAGE_FILE_NAME_FORMAT)
    mask_shape = masks[0].shape
    for i in range(len(images)):
        image = images[i]
        # overlay_mask = np.array(TEST_SEQUENCE_OVERLAY_COLOR)
        # overlay_mask = np.repeat(overlay_mask, mask_shape[0], axis=0)
        # overlay_mask = np.repeat(overlay_mask, mask_shape[1], axis=1)

        misc.imsave(frame_image_path % i, image)
    generator_utils.create_video(images_dir, os.path.join(path, VIDEO_FILE))


# Create the model
model = Sequential()
model.add(Convolution2D(32, 7, 7, activation='relu',
                        input_shape=(constants.RESOLUTION_HEIGHT, constants.RESOLUTION_WIDTH, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(64, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(Convolution2D(1, 1, 1, activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

output_shape = model.layers[-1].output_shape[1:]

# Read data
print('Reading training, validation and testing data')
start = utils.start_timer()
x_train, y_train = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.VALIDATION_DATASET_PATH),
                                    output_shape)

# # TODO remove this
# x_train = x_train[0:50]
# y_train = y_train[0:50]

x_validation, y_validation = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.VALIDATION_DATASET_PATH),
                                              output_shape)
x_test, y_test = read_toy_dataset(os.path.join(constants.OUTPUT_PATH, constants.TEST_DATASET_PATH),
                                  output_shape)
print('Data read in %.2f minutes' % utils.get_duration_minutes(start))

# Train the network
validation_results = []
for i in range(0, TRAINING_EPOCHS):
    print('Training epoch %s' % (i + 1))
    start = utils.start_timer()
    model.fit(x_train, y_train, nb_epoch=1, batch_size=BATCH_SIZE, verbose=1)
    training_duration = utils.get_duration_minutes(start)
    print('\tEpoch %s trained in %.2f minutes' % (i + 1, training_duration))

    start = utils.start_timer()
    error = model.evaluate(x_validation, y_validation, batch_size=BATCH_SIZE)
    evaluation_duration = utils.get_duration_minutes(start)
    print('\tModel evaluated in %.2f minutes' % evaluation_duration)
    print('\tValidation result - loss: %s, accuracy: %s' % tuple(error))
    validation_results.append(error)

# Evaluate performance
print("Training finished")
test_error = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print('\tTest result - loss: %s, accuracy: %s' % tuple(test_error))

# Plot validation error development
graph_x = range(1, len(validation_results) + 1)
validation_results = np.transpose(np.array(validation_results))
line = plt.plot(graph_x, validation_results[0], graph_x, validation_results[1])
plt.title('Validation error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.xticks(graph_x)
plt.grid()
plt.show()

# Generate annotated test video sequence
print('Creating annotated test video sequence')
y_predicted = model.predict(x_test, batch_size=BATCH_SIZE)
generate_video_sequence(RESULT_DIR, x_test, y_predicted)
