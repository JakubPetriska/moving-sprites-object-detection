import random

import numpy as np


def _convert_from_frame_to_mask_coordinates(coordinate, scale_factor, dimension_size):
    return min(int(coordinate * scale_factor), dimension_size - 1)


def create_masks(data_size, labels, mask_shape, allowed_object_types=None):
    """Create masks that serve as training ground truth.
    :param data_size: Size of the data in format (sample_count, sample_height, sample_width)
    :param labels: Labels of the training images.
    :param mask_shape: Shape of the masks.
    :param allowed_object_types: Types of objects that should be put into the masks.
    :return: The created masks.
    """
    sample_count = data_size[0]
    frame_shape = data_size[1:3]

    vertical_scale_factor = mask_shape[0] / frame_shape[0]
    horizontal_scale_factor = mask_shape[1] / frame_shape[1]

    y = np.zeros([sample_count] + list(mask_shape), dtype=np.uint8)
    for image_index in range(sample_count):
        frame_labels = labels[image_index]
        for object_label in frame_labels:
            object_type = object_label[0]
            if not allowed_object_types or object_type in allowed_object_types:
                top = _convert_from_frame_to_mask_coordinates(object_label[2], vertical_scale_factor, frame_shape[0])
                bottom = _convert_from_frame_to_mask_coordinates(object_label[3], vertical_scale_factor, frame_shape[0])
                left = _convert_from_frame_to_mask_coordinates(object_label[4], horizontal_scale_factor, frame_shape[1])
                right = _convert_from_frame_to_mask_coordinates(object_label[5], horizontal_scale_factor,
                                                                frame_shape[1])
                y[image_index, top:bottom + 1, left:right + 1] = 1
    return y


def split_data(x, y, validation_split, num_sets=-1):
    """Create multiple data splits into training and validation data.
    The splitting is very similar to k-fold cross validation but the requested number of sets can be lower than k,
    in which case the folds for validation data are selected randomly.
    :param x: Training images.
    :param y: Labels of training images.
    :param validation_split: Percentage of data that is used for validation.
    :param num_sets: Number of data sets needed.
    :return: The split datasets as tuples in format (x_train, y_train, x_val, y_val).
    """
    data_slice_count = int(1 / validation_split)
    if num_sets == -1:
        num_sets = data_slice_count

    if num_sets > data_slice_count:
        raise ValueError('Cannot generate more data sets than the number of validation data slices.')

    if num_sets < data_slice_count:
        validation_data_slice_indices = []
        for i in range(num_sets):
            validation_data_slice_index = -1
            while validation_data_slice_index == -1 or validation_data_slice_index in validation_data_slice_indices:
                validation_data_slice_index = random.randint(0, data_slice_count - 1)
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
