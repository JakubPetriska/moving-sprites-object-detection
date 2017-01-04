import csv
import os

import datetime
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

KITTI_DIR = os.path.join(os.pardir, os.pardir, 'kitti')
KITTI_IMAGES_DIR = os.path.join(KITTI_DIR, 'image_02')
KITTI_LABELS_DIR = os.path.join(KITTI_DIR, 'label_02')
KITTI_IMAGES_FILE_FORMAT = '%06d.png'

KITTI_ORIGINAL_RESOLUTION_WIDTH = 1242
KITTI_ORIGINAL_RESOLUTION_HEIGHT = 375
KITTI_USED_RESOLUTION_WIDTH = 400
KITTI_USED_RESOLUTION_HEIGHT = 120

TEST_SEQUENCE_OVERLAY_ALPHA = 0.25
TEST_SEQUENCE_OVERLAY_COLOR = [0, 255, 0]


def read_toy_dataset(path, mask_shape, allowed_types=None):
    labels = generator_utils.read_labels(os.path.join(path, constants.DATASET_LABELS_FILE))
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


def read_kitti_dataset(mask_shape, max_frames=-1, allowed_types=None, log=False):
    images_dirs = os.listdir(KITTI_IMAGES_DIR)

    mask_vertical_scale_factor = mask_shape[0] / KITTI_ORIGINAL_RESOLUTION_HEIGHT
    mask_horizontal_scale_factor = mask_shape[1] / KITTI_ORIGINAL_RESOLUTION_WIDTH

    if max_frames > 0:
        total_images_count = max_frames
    else:
        total_images_count = 0
        for images_dir_name in sorted(images_dirs):
            images_dir_path = os.path.join(KITTI_IMAGES_DIR, images_dir_name)
            total_images_count += len(os.listdir(images_dir_path))

    x = np.empty((total_images_count, KITTI_USED_RESOLUTION_HEIGHT, KITTI_USED_RESOLUTION_WIDTH, 3))
    y = np.empty([total_images_count] + list(mask_shape))
    image_index = 0
    for images_dir_name in sorted(images_dirs):
        if image_index >= total_images_count:
            break
        if log:
            print('%s: Reading dir %s' % (datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), images_dir_name))
        images_dir_path = os.path.join(KITTI_IMAGES_DIR, images_dir_name)
        labels_file_path = os.path.join(KITTI_LABELS_DIR, images_dir_name + '.txt')

        # First read the labels for the whole sequence, read only the allowed objects
        sequence_labels = []
        with open(labels_file_path, newline='') as labels_file:
            labels_reader = csv.reader(labels_file, delimiter=' ')
            current_frame_index = -1
            for row in labels_reader:
                frame_index = int(row[0])
                if frame_index > current_frame_index:
                    # Labels of new frame are starting
                    # It's possible that some frames were skipped since they do not contain any objects
                    for i in range(frame_index - current_frame_index):
                        sequence_labels.append([])
                    current_frame_index = frame_index

                object_type = row[2]
                if not allowed_types or object_type in allowed_types:
                    # If the object is allowed
                    box_x1 = float(row[6])
                    box_y1 = float(row[7])
                    box_x2 = float(row[8])
                    box_y2 = float(row[9])
                    sequence_labels[-1].append((object_type, box_x1, box_y1, box_x2, box_y2))

        # Now read the frames and create ground truth masks
        for frame_index in range(len(sequence_labels)):
            if image_index >= total_images_count:
                break

            image_file_path = os.path.join(images_dir_path, KITTI_IMAGES_FILE_FORMAT) % frame_index
            image = misc.imread(image_file_path)
            image = misc.imresize(image, (KITTI_USED_RESOLUTION_HEIGHT, KITTI_USED_RESOLUTION_WIDTH, 3))
            image = np.reshape(image, [1] + list(image.shape))
            x[image_index] = image

            frame_labels = sequence_labels[frame_index]
            mask = np.zeros(mask_shape)
            for object_label in frame_labels:
                box_x1 = object_label[1]
                box_y1 = object_label[2]
                box_x2 = object_label[3]
                box_y2 = object_label[4]

                object_bounding_box = [box_y1, box_y2, box_x1, box_x2]
                scaled_vertical_bounds = [min(round(bound * mask_vertical_scale_factor), mask_shape[0] - 1)
                                          for bound in object_bounding_box[:2]]
                scaled_horizontal_bounds = [min(round(bound * mask_horizontal_scale_factor), mask_shape[1] - 1)
                                            for bound in object_bounding_box[2:]]
                top = scaled_vertical_bounds[0]
                bottom = scaled_vertical_bounds[1]
                left = scaled_horizontal_bounds[0]
                right = scaled_horizontal_bounds[1]
                mask[top:bottom + 1, left:right + 1] = 1

                y[image_index] = np.reshape(mask, [1] + list(mask_shape))
            image_index += 1

    if not image_index == total_images_count:
        raise ValueError('Number of frames not matching')
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
        mask = misc.imresize(mask, image.shape)
        mask = mask.astype(np.float)
        mask /= 255
        overlay_mask = np.empty(shape=(1, 1, 3))
        overlay_mask[0, 0] = TEST_SEQUENCE_OVERLAY_COLOR
        overlay_mask = np.repeat(overlay_mask, image.shape[0], axis=0)
        overlay_mask = np.repeat(overlay_mask, image.shape[1], axis=1)
        overlay_mask *= TEST_SEQUENCE_OVERLAY_ALPHA
        overlay_mask *= mask
        image *= 1 - (mask * (1 - TEST_SEQUENCE_OVERLAY_ALPHA))
        image += overlay_mask
        misc.imsave(frame_image_path % i, image.astype(np.uint8))
    generator_utils.create_video(images_dir, output_path)
