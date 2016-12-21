import os

import numpy as np
from scipy import misc

from toy_dataset_generator import constants
from toy_dataset_generator import generator_utils

# Evaluation result folder structure
MODEL_FILE = 'model.json'
MODEL_WEIGHTS_FILE = 'model_weights.h5'
MODEL_BEST_WEIGHTS_FILE = 'model_best_weights.h5'  # weights of model with best validation performance
PREDICTED_MASKS_DIR = 'masks_predicted'
VIDEO_FILE = 'video.mp4'
IMAGES_DIR = 'images_annotated'

TEST_SEQUENCE_OVERLAY_ALPHA = 0.25
TEST_SEQUENCE_OVERLAY_COLOR = [0, 255, 0]


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
        mask_image = np.copy(masks[i])
        mask_image *= 255
        mask_image = np.clip(mask_image, 0, 255)
        mask_image = np.round(mask_image).astype(np.uint8)
        mask_image = np.repeat(mask_image, 3, axis=2)
        misc.imsave(mask_image_path % i, mask_image)


def generate_video_sequence(output_path, images_dir, images, masks):
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    frame_image_path = os.path.join(images_dir, constants.FRAME_IMAGE_FILE_NAME_FORMAT)
    for i in range(len(images)):
        image = np.copy(images[i]).astype(np.float)
        mask = np.clip(np.copy(masks[i]), 0, 1)
        mask *= 255
        mask = np.repeat(mask.astype(np.uint8), 3, axis=2)
        mask = misc.imresize(mask, (constants.RESOLUTION_HEIGHT, constants.RESOLUTION_WIDTH, 3))
        mask = mask.astype(np.float)
        mask /= 255
        overlay_mask = np.empty(shape=(1, 1, 3))
        overlay_mask[0, 0] = TEST_SEQUENCE_OVERLAY_COLOR
        overlay_mask = np.repeat(overlay_mask, constants.RESOLUTION_HEIGHT, axis=0)
        overlay_mask = np.repeat(overlay_mask, constants.RESOLUTION_WIDTH, axis=1)
        overlay_mask *= TEST_SEQUENCE_OVERLAY_ALPHA
        overlay_mask *= mask
        image *= 1 - (mask * (1 - TEST_SEQUENCE_OVERLAY_ALPHA))
        image += overlay_mask
        misc.imsave(frame_image_path % i, image.astype(np.uint8))
    generator_utils.create_video(images_dir, output_path)
