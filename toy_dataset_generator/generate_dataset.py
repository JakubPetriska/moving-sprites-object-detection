import os.path
import random
from os import listdir

import numpy as np
from scipy import misc

from toy_dataset_generator import constants
from toy_dataset_generator import generator_utils
from common import utils

SHOW_VIDEO_ENCODING_INFO_LOG = False
SHOW_TIME_LOG = True
SHOW_OBJECT_RECTANGLES = False


def apply_gaussian_noise(frame):
    noise = np.random.normal(size=frame.shape, scale=10)
    frame += noise


class Sprite:
    def __init__(self, sprite_image, initial_position, initial_scale, velocity, movement_function,
                 scale_speed, scale_function):
        self.sprite_image = sprite_image
        self.lifetime = 0
        self.initial_position = initial_position
        self.initial_scale = initial_scale
        self.velocity = velocity
        self.movement_function = movement_function
        self.scale_speed = scale_speed
        self.scale_function = scale_function

    def increase_lifetime(self):
        self.lifetime += 1 / constants.FRAMES_PER_SECOND

    def render(self, frame):
        position = np.round(self.movement_function(self.initial_position, self.lifetime, self.velocity)) \
            .astype(np.int64)
        scale = self.scale_function(self.initial_scale, self.lifetime, self.scale_speed)

        scaled_sprite_image_size \
            = (round(scale[0] * self.sprite_image.shape[0]), round(scale[1] * self.sprite_image.shape[1]))
        scaled_sprite_image_size = [int(a) for a in scaled_sprite_image_size]
        top = position[0]
        bottom = top + scaled_sprite_image_size[0]
        left = position[1]
        right = left + scaled_sprite_image_size[1]
        if bottom < 0 or top >= constants.RESOLUTION_HEIGHT \
                or right < 0 or left >= constants.RESOLUTION_WIDTH:
            return None
        else:
            scaled_sprite = misc.imresize(self.sprite_image, scaled_sprite_image_size)
            # Take only the visible part of sprite
            overlap_top = max(top * -1, 0)
            overlap_bottom = max(bottom - constants.RESOLUTION_HEIGHT + 1, 0)
            overlap_left = max(left * -1, 0)
            overlap_right = max(right - constants.RESOLUTION_WIDTH + 1, 0)
            scaled_sprite \
                = scaled_sprite[overlap_top:scaled_sprite_image_size[0] - overlap_bottom,
                  overlap_left:scaled_sprite_image_size[1] - overlap_right, :]
            position = np.clip(position,
                               0, max(constants.RESOLUTION_HEIGHT, constants.RESOLUTION_WIDTH))
            sprite_alpha = scaled_sprite[:, :, 3] / 255
            background_alpha = -sprite_alpha + 1
            for i in range(0, 3):
                frame[position[0]:position[0] + scaled_sprite.shape[0],
                position[1]:position[1] + scaled_sprite.shape[1], i] \
                    *= background_alpha
                frame[position[0]:position[0] + scaled_sprite.shape[0],
                position[1]:position[1] + scaled_sprite.shape[1], i] \
                    += scaled_sprite[:, :, i] * sprite_alpha

            if SHOW_OBJECT_RECTANGLES:
                frame[top:bottom, left:left + 1, 0] = 255
                frame[top:bottom, left:left + 1, 1:] = 0
                frame[top:bottom, right:right + 1, 0] = 255
                frame[top:bottom, right:right + 1, 1:] = 0
                frame[top:top + 1, left:right, 0] = 255
                frame[top:top + 1, left:right, 1:] = 0
                frame[bottom:bottom + 1, left:right, 0] = 255
                frame[bottom:bottom + 1, left:right, 1:] = 0
            return 0 if overlap_top > 0 else top, bottom - overlap_bottom, \
                   0 if overlap_left > 0 else left, right - overlap_right


class SequenceGenerator:
    def __init__(self, sprite_images):
        self.sprite_images = sprite_images
        self.sprites = []
        self.movement_functions = []
        self.scale_functions = []

        # Linear movement
        self.movement_functions.append(
            lambda initial_position, lifetime, velocity: initial_position + velocity * lifetime)
        # Linear scale with possible shearing
        self.scale_functions.append(
            lambda initial_scale, lifetime, scale_speed: np.clip(initial_scale + scale_speed * lifetime,
                                                                 constants.SPRITE_MIN_SCALE, 1))

    def _spawn_sprite(self):
        sprite_image = random.choice(self.sprite_images)
        half_shape = np.array(sprite_image.shape) / 2
        initial_position = np.array((random.randrange(0, constants.RESOLUTION_HEIGHT) - half_shape[0],
                                     random.randrange(0, constants.RESOLUTION_WIDTH) - half_shape[1]))
        velocity = np.array((random.random() - 0.5, random.random() - 0.5))
        velocity /= np.linalg.norm(velocity)
        velocity *= random.gauss(mu=constants.MEAN_SPRITE_MOVEMENT_SPEED,
                                 sigma=constants.MEAN_SPRITE_MOVEMENT_SPEED / 2)

        if constants.ALLOW_SPRITE_SHEARING:
            initial_scale = np.array([random.uniform(constants.SPRITE_MIN_SCALE, 1) for i in range(0, 2)])
            scale_speed = np.array(
                [random.gauss(mu=constants.MEAN_SPRITE_SCALE_SPEED, sigma=constants.MEAN_SPRITE_SCALE_SPEED / 2)
                 for i in range(0, 2)])
        else:
            initial_scale = np.array([random.uniform(constants.SPRITE_MIN_SCALE, 0.8)] * 2)
            scale_speed = np.array([random.gauss(mu=constants.MEAN_SPRITE_SCALE_SPEED,
                                                 sigma=constants.MEAN_SPRITE_SCALE_SPEED / 2)] * 2)
        if random.random() >= 0.5:
            scale_speed *= -1
        return Sprite(sprite_image, initial_position, initial_scale,
                      velocity, random.choice(self.movement_functions),
                      scale_speed, random.choice(self.scale_functions))

    def _spawn_probability(self):
        zero_sprite_spawn_probability = 0.5
        slope = (0.2 - zero_sprite_spawn_probability) / constants.AVERAGE_SPRITE_COUNT
        return slope * len(self.sprites) + zero_sprite_spawn_probability

    def generate_next_frame(self):
        for sprite in self.sprites:
            sprite.increase_lifetime()

        if random.random() <= self._spawn_probability():
            self.sprites.append(self._spawn_sprite())

        frame = np.ones((constants.RESOLUTION_HEIGHT, constants.RESOLUTION_WIDTH, 3))
        frame *= 210
        apply_gaussian_noise(frame)

        frame_labels = []
        for i in range(len(self.sprites) - 1, -1, -1):
            sprite_boundaries = self.sprites[i].render(frame)
            if not sprite_boundaries:
                self.sprites.pop(i)
            else:
                frame_labels.append(sprite_boundaries)

        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame, frame_labels


def generate_sequence(frame_count, folder_path):
    if frame_count <= 0:
        return

    images_dir = os.path.join(folder_path, constants.DATASET_IMAGES_DIR)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    sprite_images = []
    for sprite_file_name in listdir(constants.SPRITES_DIR):
        image = misc.imread(os.path.join(constants.SPRITES_DIR, sprite_file_name))
        scale_factor = constants.RESOLUTION_WIDTH / image.shape[1]
        scaled_shape = [int(d * scale_factor) for d in image.shape]
        scaled_shape[2:] = image.shape[2:]
        image = misc.imresize(image, scaled_shape)
        sprite_images.append(image)

    # Generate the video frames
    sequence_generator = SequenceGenerator(sprite_images)
    frame_generation_start = utils.start_timer()
    frame_image_path_format = os.path.join(images_dir, constants.FRAME_IMAGE_FILE_NAME_FORMAT)
    sequence_labels = []
    for i in range(0, frame_count):
        frame, frame_labels = sequence_generator.generate_next_frame()
        sequence_labels.append([i] + frame_labels)
        misc.imsave(frame_image_path_format % i, frame)

    frames_generation_duration = utils.get_duration_secs(frame_generation_start)
    print('\tFrames generated')
    if SHOW_TIME_LOG:
        print('\t\tGenerated %d frames in %.1f seconds, average time per frame is %f seconds'
              % (frame_count, frames_generation_duration, frames_generation_duration / frame_count))

    # Generate video file
    video_encoding_start = utils.start_timer()
    generator_utils.create_video(images_dir, os.path.join(folder_path, constants.DATASET_VIDEO_FILE),
                                 SHOW_VIDEO_ENCODING_INFO_LOG)
    print('\tVideo file generated')
    if SHOW_TIME_LOG:
        print('\t\tGenerated video in %.1f seconds' % utils.get_duration_secs(video_encoding_start))

    generator_utils.write_labels(os.path.join(folder_path, constants.DATASET_LABELS_FILE), sequence_labels)
    print('\tLabels saved')

    annotation_start = utils.start_timer()
    generator_utils.annotate_dataset(images_dir, os.path.join(folder_path, constants.DATASET_LABELS_FILE),
                                     os.path.join(folder_path, constants.DATASET_IMAGES_ANNOTATED_DIR),
                                     os.path.join(folder_path, constants.DATASET_VIDEO_ANNOTATED_FILE),
                                     SHOW_VIDEO_ENCODING_INFO_LOG)
    print('\tAnnotated data created')
    if SHOW_TIME_LOG:
        print('\t\tAnnotated dataset in %.1f seconds' % utils.get_duration_secs(annotation_start))


print('Generating training data')
generate_sequence(constants.FRAME_COUNT_TRAINING,
                  os.path.join(constants.OUTPUT_PATH, constants.TRAINING_DATASET_PATH))
print('Generating validation data')
generate_sequence(constants.FRAME_COUNT_VALIDATION,
                  os.path.join(constants.OUTPUT_PATH, constants.VALIDATION_DATASET_PATH))
print('Generating test data')
generate_sequence(constants.FRAME_COUNT_TEST,
                  os.path.join(constants.OUTPUT_PATH, constants.TEST_DATASET_PATH))
