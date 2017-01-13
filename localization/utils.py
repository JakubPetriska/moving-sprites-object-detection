import os

import numpy as np
from scipy import misc

FRAME_IMAGE_FILE_NAME_FORMAT = '%06d.png'

TEST_SEQUENCE_OVERLAY_ALPHA = 0.25
TEST_SEQUENCE_OVERLAY_COLOR = [0, 255, 0]

FRAMES_PER_SECOND = 8


def save_masks(path, masks):
    """
    Save masks either generated, or ground truth into the directory on given path as image files.
    :param path: Path of folder to which the masks will be saved.
    :param masks: Masks to save
    """
    if not os.path.exists(path):
        os.makedirs(path)
    mask_image_path = os.path.join(path, FRAME_IMAGE_FILE_NAME_FORMAT)
    for i in range(len(masks)):
        mask_image = np.copy(masks[i])
        mask_image *= 255
        mask_image = np.clip(mask_image, 0, 255)
        mask_image = np.round(mask_image).astype(np.uint8)
        mask_image = np.repeat(mask_image, 3, axis=2)
        misc.imsave(mask_image_path % i, mask_image)


def create_video(images_dir, output_file_path, show_encoding_info=False):
    """
    Generate video file from images in given directory.
    The images in the directory must have the file name format specified by constants.FRAME_IMAGE_FILE_NAME_FORMAT.
    :param images_dir: Directory in which the frame images for video are saved.
    :param output_file_path: Output file path at which the video file will be saved.
    :param show_encoding_info: Whether additional information about the encoding should be shown.
    """
    os.system('ffmpeg -f image2 -r %d -i %s -loglevel %s -vcodec mpeg4 -y %s'
              % (FRAMES_PER_SECOND, os.path.join(images_dir, FRAME_IMAGE_FILE_NAME_FORMAT),
                 '32' if show_encoding_info else '24',
                 output_file_path))


def generate_video_frames(images, masks, images_dir):
    """
    Generate frames of annotated video sequence from given input images and their masks.
    :param images: Input images.
    :param masks: Masks of the input images.
    :param images_dir: Directory in which the video frames images will be saved.
    """
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    frame_image_path = os.path.join(images_dir, FRAME_IMAGE_FILE_NAME_FORMAT)
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
