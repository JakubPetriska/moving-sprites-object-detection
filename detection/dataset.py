import csv
import os
from ast import literal_eval as make_tuple

import numpy as np
from scipy import misc
from detection import constants

LABELS_FILE_DELIMITER = ';'


def read_labels(input_path):
    labels = []
    with open(input_path, newline='') as labels_file:
        labels_reader = csv.reader(labels_file, delimiter=LABELS_FILE_DELIMITER)
        for row in labels_reader:
            row[0] = int(row[0])
            for i in range(1, len(row)):
                row[i] = make_tuple(row[i])
                pass
            labels.append(row)
    return labels


def read_dataset(path, mask_shape, allowed_types=None):
    labels = read_labels(os.path.join(path, constants.DATASET_LABELS_FILE))
    image_path_format = os.path.join(path, constants.DATASET_IMAGES_DIR, constants.FRAME_IMAGE_FILE_NAME_FORMAT)
    x = np.empty((len(labels), constants.RESOLUTION_HEIGHT, constants.RESOLUTION_WIDTH, 3))
    y_shape = [len(labels)]
    y_shape += mask_shape
    y = np.empty(y_shape)

    vertical_scale_factor = mask_shape[0] / constants.RESOLUTION_HEIGHT
    horizontal_scale_factor = mask_shape[1] / constants.RESOLUTION_WIDTH
    for i in range(0, len(labels)):
        label = labels[i]
        image = misc.imread(image_path_format % label[0])
        mask = np.zeros(mask_shape)
        for object_label in label[1:]:
            object_type = object_label[0]
            if allowed_types and object_type not in allowed_types:
                # If object is not allowed don't put it into ground truth mask
                continue
            object_bounding_box = object_label[1]
            scaled_vertical_bounds = [min(round(bound * vertical_scale_factor), mask_shape[0] - 1)
                                      for bound in object_bounding_box[:2]]
            scaled_horizontal_bounds = [min(round(bound * horizontal_scale_factor), mask_shape[1] - 1)
                                        for bound in object_bounding_box[2:]]
            top = scaled_vertical_bounds[0]
            bottom = scaled_vertical_bounds[1]
            left = scaled_horizontal_bounds[0]
            right = scaled_horizontal_bounds[1]
            mask[top:bottom + 1, left:right + 1] = 1
        x[i] = image
        y[i] = mask
    return x, y
