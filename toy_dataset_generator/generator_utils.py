import csv
import os
from ast import literal_eval as make_tuple
from scipy import misc

from toy_dataset_generator import constants

LABELS_FILE_DELIMITER = ';'


def create_video(images_dir, output_file_path, show_encoding_info=False):
    os.system('ffmpeg -f image2 -r %d -i %s -loglevel %s -vcodec mpeg4 -y %s'
              % (constants.FRAMES_PER_SECOND, os.path.join(images_dir, constants.FRAME_IMAGE_FILE_NAME_FORMAT),
                 '32' if show_encoding_info else '24',
                 output_file_path))


def write_labels(output_path, labels):
    with open(output_path, 'w', newline='') as labels_file:
        labels_writer = csv.writer(labels_file, delimiter=LABELS_FILE_DELIMITER)
        for frame_labels in labels:
            labels_writer.writerow(frame_labels)


def read_labels(input_path):
    labels = []
    with open(input_path, newline='') as labels_file:
        labels_reader = csv.reader(labels_file, delimiter=LABELS_FILE_DELIMITER)
        for row in labels_reader:
            row[0] = int(row[0])
            for i in range(1,len(row)):
                row[i] = make_tuple(row[i])
                pass
            labels.append(row)
    return labels


def annotate_dataset(dataset_images_dir, labels_file_path, output_annotated_images_dir, output_video_path,
                     show_encoding_info, allowed_types=None):
    labels = read_labels(labels_file_path)
    if not os.path.exists(output_annotated_images_dir):
        os.makedirs(output_annotated_images_dir)

    dataset_frame_image_path = os.path.join(dataset_images_dir, constants.FRAME_IMAGE_FILE_NAME_FORMAT)
    output_frame_image_path = os.path.join(output_annotated_images_dir, constants.FRAME_IMAGE_FILE_NAME_FORMAT)
    for frame_labels in labels:
        frame_index = frame_labels[0]
        frame = misc.imread(dataset_frame_image_path % frame_index)
        for i in range(1, len(frame_labels)):
            object_label = frame_labels[i]
            object_type = object_label[0]
            if allowed_types and object_type not in allowed_types:
                # If object is not allowed skip drawing it's frame
                continue
            object_bounding_box = object_label[1]
            top = object_bounding_box[0]
            bottom = object_bounding_box[1]
            left = object_bounding_box[2]
            right = object_bounding_box[3]

            frame[top:bottom, left, 0] = 255
            frame[top:bottom, left, 1:] = 0
            frame[top:bottom, right, 0] = 255
            frame[top:bottom, right, 1:] = 0
            frame[top, left:right, 0] = 255
            frame[top, left:right, 1:] = 0
            frame[bottom, left:right, 0] = 255
            frame[bottom, left:right, 1:] = 0

        misc.imsave(output_frame_image_path % frame_index, frame)

    create_video(output_annotated_images_dir, output_video_path, show_encoding_info)
