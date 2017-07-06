import os.path

# Paths and dataset folder structure
DATASET_PATH = os.path.join(os.pardir, 'moving-sprites-dataset', 'dataset')
DATASET_TRAINING_PATH = os.path.join(DATASET_PATH, 'training')
DATASET_VALIDATION_PATH = os.path.join(DATASET_PATH, 'validation')
DATASET_TEST_PATH = os.path.join(DATASET_PATH, 'test')

DATASET_IMAGES_DIR = 'images'
DATASET_LABELS_FILE = 'labels.csv'
FRAME_IMAGE_FILE_NAME_FORMAT = 'image%05d.png'

# Dataset properties
RESOLUTION_WIDTH = 200
RESOLUTION_HEIGHT = 200
FRAMES_PER_SECOND = 8
