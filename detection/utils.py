import os

import numpy as np
from scipy import misc

from detection import constants

TEST_SEQUENCE_OVERLAY_ALPHA = 0.25
TEST_SEQUENCE_OVERLAY_COLOR = [0, 255, 0]


def save_mask(path, mask):
    mask_image = np.copy(mask)
    mask_image *= 255
    mask_image = np.clip(mask_image, 0, 255)
    mask_image = np.round(mask_image).astype(np.uint8)
    mask_image = np.repeat(mask_image, 3, axis=2)
    misc.imsave(path, mask_image)


def save_mask_colored(path, mask):
    mask_image = np.copy(mask)
    mask_image *= 255
    mask_image = np.clip(mask_image, 0, 255)
    mask_image = np.round(mask_image).astype(np.uint8)
    misc.imsave(path, mask_image)


def save_masks(path, masks):
    if not os.path.exists(path):
        os.makedirs(path)
    mask_image_path = os.path.join(path, constants.FRAME_IMAGE_FILE_NAME_FORMAT)
    for i in range(len(masks)):
        save_mask(mask_image_path % i, masks[i])


def create_video(images_dir, output_file_path, show_encoding_info=False):
    os.system('ffmpeg -f image2 -r %d -i %s -loglevel %s -vcodec mpeg4 -y %s'
              % (constants.FRAMES_PER_SECOND, os.path.join(images_dir, constants.FRAME_IMAGE_FILE_NAME_FORMAT),
                 '32' if show_encoding_info else '24',
                 output_file_path))


def generate_video_sequence(output_file_path, images_dir, images, masks):
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
    create_video(images_dir, output_file_path)


def annotate_image(image, cluster_bounds):
    for i in range(len(cluster_bounds)):
        object_bounding_box = cluster_bounds[i]
        top = object_bounding_box[0]
        bottom = object_bounding_box[1]
        left = object_bounding_box[2]
        right = object_bounding_box[3]

        image[top:bottom, left, 0] = 255
        image[top:bottom, left, 1:] = 0
        image[top:bottom, right, 0] = 255
        image[top:bottom, right, 1:] = 0
        image[top, left:right, 0] = 255
        image[top, left:right, 1:] = 0
        image[bottom, left:right, 0] = 255
        image[bottom, left:right, 1:] = 0
