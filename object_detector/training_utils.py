import datetime
import os

from keras.callbacks import TensorBoard, ModelCheckpoint
from tabulate import tabulate

from object_detector import utils
from object_detector.utils import generate_video_sequence
from object_detector.utils import save_masks

OUTPUT_DIR = 'result_%s'
TENSORBOARD_LOGS_DIR = 'tensorboard_logs'
MODEL_PLOT = 'model.png'
MASKS_DIR = 'masks_ground_truth'
OUTPUT_INFO_FILE = 'output'

LIGHT_OUTPUT = True  # Turns off all following
SAVE_GROUND_TRUTH_TEST_MASKS = True
SAVE_PREDICTED_TEST_MASKS = True
GENERATE_ANNOTATED_VIDEO = True


def train_and_evaluate(model_wrapper, x_train, y_train, x_validation, y_validation, x_test, y_test,
                       verbosity=1, plot_model=True, results_dir='results'):
    output_dir_name = OUTPUT_DIR % datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    print('Output dir: %s' % output_dir_name)
    output_dir = os.path.join(os.pardir, os.pardir, results_dir, '%s') % output_dir_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Do initial evaluation of validation and test data
    initial_training_eval = model_wrapper.evaluate(x_train, y_train)
    initial_validation_eval = model_wrapper.evaluate(x_validation, y_validation)
    initial_test_eval = model_wrapper.evaluate(x_test, y_test)

    # Train the network
    tensorboard_callback = TensorBoard(log_dir=os.path.join(output_dir, TENSORBOARD_LOGS_DIR))
    model_checkpoint_callback = ModelCheckpoint(os.path.join(output_dir, utils.MODEL_BEST_WEIGHTS_FILE),
                                                monitor='val_acc', verbose=verbosity, save_best_only=True,
                                                save_weights_only=True)
    model_wrapper.train(x_train, y_train, validation_data=(x_validation, y_validation),
                        callbacks=[tensorboard_callback, model_checkpoint_callback])

    # Save model
    print('Saving model to disk')
    model_wrapper.save_to_disk(os.path.join(output_dir, utils.MODEL_FILE),
                               os.path.join(output_dir, utils.MODEL_WEIGHTS_FILE))

    # Evaluate performance
    print("Training finished")
    final_training_eval = model_wrapper.evaluate(x_train, y_train)
    final_validation_eval = model_wrapper.evaluate(x_validation, y_validation)
    final_test_eval = model_wrapper.evaluate(x_test, y_test)

    # Generate annotated test video sequence
    if not LIGHT_OUTPUT:
        print('Creating annotated test data')
        if SAVE_GROUND_TRUTH_TEST_MASKS:
            save_masks(os.path.join(output_dir, MASKS_DIR), y_test)
        if SAVE_PREDICTED_TEST_MASKS or GENERATE_ANNOTATED_VIDEO:
            y_predicted = model_wrapper.predict(x_test)
            if SAVE_PREDICTED_TEST_MASKS:
                save_masks(os.path.join(output_dir, utils.PREDICTED_MASKS_DIR), y_predicted)
            if GENERATE_ANNOTATED_VIDEO:
                generate_video_sequence(os.path.join(output_dir, utils.VIDEO_FILE),
                                        os.path.join(output_dir, utils.IMAGES_DIR),
                                        x_test, y_predicted)

    # Plot the model
    if plot_model:
        from keras.utils.visualize_util import plot
        plot(model_wrapper.model, to_file=os.path.join(output_dir, MODEL_PLOT), show_shapes=True)

    evaluation_table = tabulate([['Training', initial_training_eval[0], initial_training_eval[1],
                                  final_training_eval[0], final_training_eval[1]],
                                 ['Validation', initial_validation_eval[0], initial_validation_eval[1],
                                  final_validation_eval[0], final_validation_eval[1]],
                                 ['Test', initial_test_eval[0], initial_test_eval[1],
                                  final_test_eval[0], final_test_eval[1]]],
                                headers=['Data', 'Initial loss', 'Initial accuracy', 'Final loss', 'Final accuracy'])
    with open(os.path.join(output_dir, OUTPUT_INFO_FILE), mode='w') as output_file:
        output_file.write(evaluation_table)
    print('\n' + evaluation_table)
