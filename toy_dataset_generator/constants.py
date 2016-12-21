import os.path

# Paths
OUTPUT_PATH = os.path.join(os.pardir, os.pardir, 'toy_dataset')
SPRITES_DIR = os.path.join(os.pardir, os.pardir, 'sprites')

# Output data paths and structure
DATASET_IMAGES_DIR = 'images'
DATASET_IMAGES_ANNOTATED_DIR = 'images_annotated'
DATASET_VIDEO_FILE = 'video.mp4'
DATASET_VIDEO_ANNOTATED_FILE = 'video_annotated.mp4'
DATASET_LABELS_FILE = 'labels.txt'
FRAME_IMAGE_FILE_NAME_FORMAT = 'image%05d.png'

TRAINING_DATASET_PATH = 'training'
VALIDATION_DATASET_PATH = 'validation'
TEST_DATASET_PATH = 'test'

# Dataset properties
RESOLUTION_WIDTH = 200
RESOLUTION_HEIGHT = 200
FRAMES_PER_SECOND = 8

FRAME_COUNT_TRAINING = 9000
FRAME_COUNT_VALIDATION = 1000
FRAME_COUNT_TEST = 1000

# Sprite movement
AVERAGE_SPRITE_COUNT = 2
SPRITE_MIN_SCALE = 0.1

ALLOW_SPRITE_SHEARING = False

MEAN_SPRITE_MOVEMENT_SPEED = RESOLUTION_WIDTH / 7  # Pixels per second
MEAN_SPRITE_SCALE_SPEED = 0.01  # Absolute scale per second
