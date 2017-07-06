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

# Training result folder structure
RESULTS_DIR = 'models'
RESULT_DIR_FORMAT = 'model_%s'
RESULT_MODEL_FILE = 'model.json'
RESULT_MODEL_WEIGHTS_FILE = 'model_weights.h5'
RESULT_MODEL_BEST_WEIGHTS_FILE = 'model_best_weights.h5'  # weights of model with best validation performance
RESULT_PREDICTED_MASKS_DIR = 'masks_predicted'
RESULT_VIDEO_FILE = 'video.mp4'
RESULT_IMAGES_DIR = 'images_annotated'
RESULT_MODEL_PLOT = 'model.png'
RESULT_MASKS_DIR = 'masks_ground_truth'
