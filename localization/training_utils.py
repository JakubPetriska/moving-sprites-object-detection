import random

import numpy as np


def create_masks(x, labels, mask_shape, allowed_objects=None):
    """Create masks that serve as training ground truth.
    :param x: Training images.
    :param labels: Labels of the training images.
    :param mask_shape: Shape of the masks.
    :param allowed_objects: Types of objects that should be put into the masks.
    :return: The created masks.
    """
    # TODO test
    bounds_y_scale_factor = mask_shape[0] / x.shape[1]
    bounds_x_scale_factor = mask_shape[1] / x.shape[2]

    y = np.empty([x.shape[0]] + list(mask_shape))
    for image_index in range(x.shape[0]):
        frame_labels = labels[image_index]
        mask = np.zeros(mask_shape)
        for object_label in frame_labels:
            object_type = object_label[0]
            if not allowed_objects or object_type in allowed_objects:
                box_x1 = round(object_label[1] * bounds_x_scale_factor)
                box_y1 = round(object_label[2] * bounds_y_scale_factor)
                box_x2 = round(object_label[3] * bounds_x_scale_factor)
                box_y2 = round(object_label[4] * bounds_y_scale_factor)
                mask[box_y1:box_y2 + 1, box_x1:box_x2 + 1] = 1
                y[image_index] = np.reshape(mask, [1] + list(mask_shape))
    return y


def split_data(x, y, validation_split, num_sets):
    """Create multiple data splits into training and validation data.
    The splitting is very similar to k-fold cross validation but the requested number of sets can be lower than k,
    in which case the folds for validation data are selected randomly.
    :param x: Training images.
    :param y: Labels of training images.
    :param validation_split: Percentage of data that is used for validation.
    :param num_sets: Number of data sets needed.
    :return: The split datasets.
    """
    # TODO test
    data_slice_count = int(1 / validation_split)

    if num_sets > data_slice_count:
        raise ValueError('Cannot generate more data sets than the number of validation data slices.')

    if num_sets < data_slice_count:
        validation_data_slice_indices = []
        for i in range(num_sets):
            validation_data_slice_index = -1
            while validation_data_slice_index == -1 or validation_data_slice_index in validation_data_slice_indices:
                validation_data_slice_index = random.randint(0, data_slice_count)
            validation_data_slice_indices.append(validation_data_slice_index)
    else:
        validation_data_slice_indices = range(num_sets)

    samples_count = x.shape[0]
    validation_samples_count = round(samples_count * validation_split)

    data_sets = []
    for i in range(num_sets):
        validation_data_start = validation_data_slice_indices[i] * validation_samples_count
        validation_data_end = validation_data_start + validation_samples_count

        x_train = np.concatenate((x[0:validation_data_start], x[validation_data_end:]))
        y_train = np.concatenate((y[0:validation_data_start], y[validation_data_end:]))
        x_val = x[validation_data_start:validation_data_end]
        y_val = y[validation_data_start:validation_data_end]
        data_sets.append((x_train, y_train, x_val, y_val))
    return data_sets
