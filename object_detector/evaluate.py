import os
import sys

from object_detector import utils
from object_detector.model import ToyModel
from object_detector.utils import generate_video_sequence
from object_detector.utils import read_toy_dataset
from object_detector.utils import save_masks

if not (len(sys.argv) == 3 or len(sys.argv) == 4):
    print('Usage: evaluate.py path/to/results/dir path/to/dataset_dir path/to/output_dir')
    print('\tThe output_dir is optional, in case it\'s not provided only evaluation\n'
          '\tresults will be printed, otherwise annotated frames and video will be generated.')
    sys.exit(1)

path_to_results_dir = sys.argv[1]
dataset_dir = sys.argv[2]

model_wrapper = ToyModel(model_file_path=os.path.join(path_to_results_dir, utils.MODEL_FILE),
                         weights_file_path=os.path.join(path_to_results_dir, utils.MODEL_WEIGHTS_FILE))
model_output_shape = model_wrapper.model.layers[-1].output_shape[1:]

x, y = read_toy_dataset(dataset_dir, model_output_shape)

evaluation = model_wrapper.evaluate(x, y)

if len(sys.argv) == 4:
    output_dir = sys.argv[3]

    y_predicted = model_wrapper.predict(x)
    save_masks(os.path.join(output_dir, utils.PREDICTED_MASKS_DIR), y_predicted)
    generate_video_sequence(os.path.join(output_dir, utils.VIDEO_FILE), os.path.join(output_dir, utils.IMAGES_DIR),
                            x, y_predicted)

for i in range(len(model_wrapper.model.metrics_names)):
    print('%s: %s' % (model_wrapper.model.metrics_names[i], evaluation[i]))
