import csv
import datetime
import os

import numpy as np
from scipy import misc

KITTI_DIR = os.path.join(os.pardir, os.pardir, 'kitti')
KITTI_IMAGES_DIR = os.path.join(KITTI_DIR, 'training', 'image_02')
KITTI_LABELS_DIR = os.path.join(KITTI_DIR, 'training', 'label_02')
KITTI_IMAGES_FILE_FORMAT = '%06d.png'

KITTI_RESOLUTION_WIDTH = 1242
KITTI_RESOLUTION_HEIGHT = 375


def read_kitti_dataset(image_size=None, max_frames=-1, log=False):
    """Reads images of KITTI dataset and their appropriate labels.

    The list of labels contains list for each frame. The list for given frame contains tuples with each
    tuple representing one object in the frame. The tuple syntax is as follows:
        (object_type, object_id, top, bottom, left, right)
    where bounds are coordinates of pixels on which object bounding boxes lay. So right is coordinate of
    right bounding box side as indexed from the left side of the image. Vertical coordinates are relative to top edge.
    Note that the boundaries are floats.
    :param image_size: Size into which the images should scaled after as tuple (height, width).
            If None images will be kept at their original resolution. Defaults to None.
    :param max_frames: Max frames to be read. If <=0 all images will be read. Defaults to -1.
    :param log: True if reading progress should be logged. Defaults to False.
    :return: The input images scaled into desired size and list of labels for each frame.
    """
    if not image_size:
        image_size = (KITTI_RESOLUTION_HEIGHT, KITTI_RESOLUTION_WIDTH)

    images_dirs = os.listdir(KITTI_IMAGES_DIR)

    bounds_y_scale_factor = image_size[0] / KITTI_RESOLUTION_HEIGHT
    bounds_x_scale_factor = image_size[1] / KITTI_RESOLUTION_WIDTH

    if max_frames > 0:
        total_images_count = max_frames
    else:
        total_images_count = 0
        for images_dir_name in sorted(images_dirs):
            images_dir_path = os.path.join(KITTI_IMAGES_DIR, images_dir_name)
            total_images_count += len(os.listdir(images_dir_path))

    x = np.empty((total_images_count, image_size[0], image_size[1], 3))
    labels = []

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
                object_id = row[3]
                box_x1 = min(float(row[6]) * bounds_x_scale_factor, image_size[1])
                box_y1 = min(float(row[7]) * bounds_y_scale_factor, image_size[0])
                box_x2 = min(float(row[8]) * bounds_x_scale_factor, image_size[1])
                box_y2 = min(float(row[9]) * bounds_y_scale_factor, image_size[0])
                sequence_labels[-1].append((object_type, object_id, box_y1, box_y2, box_x1, box_x2))

        # Now read the frames and append their labels into the result labels list
        for frame_index in range(len(sequence_labels)):
            if image_index >= total_images_count:
                break

            image_file_path = os.path.join(images_dir_path, KITTI_IMAGES_FILE_FORMAT) % frame_index
            image = misc.imread(image_file_path)
            image = misc.imresize(image, (image_size[0], image_size[1], 3))
            image = np.reshape(image, [1] + list(image.shape))
            x[image_index] = image

            frame_labels = sequence_labels[frame_index]
            labels.append(frame_labels)

            image_index += 1

    if not image_index == total_images_count:
        raise ValueError('Number of frames not matching')
    return x, labels
