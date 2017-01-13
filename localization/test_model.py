import csv
import datetime
import os

from keras.callbacks import History

from localization.kitti import read_kitti_dataset
from localization.model import KittiModel
from localization.model import PARAM_INPUT_HEIGHT
from localization.model import PARAM_INPUT_WIDTH
from localization.training_utils import create_masks, split_data
from localization.utils import save_masks, generate_video_frames, create_video

# Training data related settings
KITTI_USED_RESOLUTION_WIDTH = 400
KITTI_USED_RESOLUTION_HEIGHT = 120
VALIDATION_DATA_SPLIT = 0.1

# Objects that will be localized
ALLOWED_OBJECT_TYPES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']

# Debug and log settings
DEBUG = True
PROGRESS_VERBOSITY = 1
PLOT_MODEL = True
LIGHT_OUTPUT = False  # Turns off all following if true
SAVE_PREDICTED_TEST_MASKS = True
GENERATE_ANNOTATED_IMAGES = True
GENERATE_ANNOTATED_VIDEO = True

max_frames = 100 if DEBUG else -1
num_runs = 1 if DEBUG else 4

test_output_dir = 'results_%s' % datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print('\nTest output dir: %s' % test_output_dir)
test_output_dir = os.path.join(os.pardir, os.pardir, 'results', test_output_dir)
os.makedirs(test_output_dir)

model_params = {PARAM_INPUT_HEIGHT: KITTI_USED_RESOLUTION_HEIGHT, PARAM_INPUT_WIDTH: KITTI_USED_RESOLUTION_WIDTH}
model_wrapper = KittiModel(verbosity=PROGRESS_VERBOSITY, params=model_params)
mask_shape = model_wrapper.model.layers[-1].output_shape[1:]

# Plot the model
if PLOT_MODEL:
    from keras.utils.visualize_util import plot

    plot(model_wrapper.model, to_file=os.path.join(test_output_dir, 'model.png'), show_shapes=True)

# Read data
x, labels = read_kitti_dataset((KITTI_USED_RESOLUTION_HEIGHT, KITTI_USED_RESOLUTION_WIDTH), max_frames=max_frames)
y = create_masks(x, labels, mask_shape)

# Split data into multiple training/validation sets for multiple runs
data_sets = split_data(x, y, VALIDATION_DATA_SPLIT, num_runs)
for i in range(num_runs):
    x_train, y_train, x_val, y_val = data_sets[i]

    # Create an instance of the model
    model_wrapper = KittiModel(verbosity=PROGRESS_VERBOSITY, params=model_params)

    # Create directory for this run
    run_output_dir = os.path.join(test_output_dir, 'run_%d' % (i + 1))
    os.makedirs(run_output_dir)

    # Train the network
    hist = History()
    model_wrapper.train(x_train, y_train, validation_data=(x_val, y_val), callbacks=[hist])

    # Save error progress during training
    with open(os.path.join(run_output_dir, 'metrics.csv'), 'w', newline='') as metrics_file:
        metrics_writer = csv.writer(metrics_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for metric_name, values in hist.history.items():
            metrics_writer.writerow([metric_name] + values)

    # Generate images of predicted masks, annotated images and video
    if not LIGHT_OUTPUT:
        if SAVE_PREDICTED_TEST_MASKS or GENERATE_ANNOTATED_IMAGES or GENERATE_ANNOTATED_VIDEO:
            y_val_predicted = model_wrapper.predict(x_val)
            if SAVE_PREDICTED_TEST_MASKS:
                save_masks(os.path.join(run_output_dir, 'predicted_masks'), y_val_predicted)
            annotated_images_dir = os.path.join(run_output_dir, 'annotated_images')
            if GENERATE_ANNOTATED_IMAGES:
                generate_video_frames(x_val, y_val_predicted, annotated_images_dir)
            if GENERATE_ANNOTATED_VIDEO:
                create_video(annotated_images_dir, os.path.join(run_output_dir, 'validation.mp4'))
