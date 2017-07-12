import os
import sys

import numpy as np
from scipy import misc

from detection import constants
from detection.model import Model
from detection.utils import save_mask

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python detect.py <model-dir> <output-dir> <input-files>")
        sys.exit(1)

    model_dir = sys.argv[1]
    model_file_path = os.path.join(model_dir, 'model.json')
    model_wights_file_path = \
        os.path.join(model_dir, 'model_best_weights.h5')
    model = Model(model_file_path, model_wights_file_path)

    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_files = sys.argv[3:]

    x = np.empty((len(input_files),
                  constants.RESOLUTION_HEIGHT,
                  constants.RESOLUTION_WIDTH, 3))
    for i, input_file_path in enumerate(input_files):
        x[i] = misc.imread(input_file_path)

    y = model.predict(x)

    for i, input_file_path in enumerate(input_files):
        input_file_name = os.path.split(input_file_path)[1]
        output_file_path = os.path.join(output_dir, input_file_name)
        save_mask(output_file_path, y[i])
