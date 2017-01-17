import inspect
import random
import unittest

import numpy as np

from localization.training_utils import split_data, create_masks


class MaskCreationTests(unittest.TestCase):
    @staticmethod
    def _generate_direction_bounds(dimension_length):
        smaller, bigger = 0, 0
        while smaller == bigger:
            smaller, bigger = [random.randint(0, dimension_length - 1) for i in range(2)]
        if smaller > bigger:
            smaller, bigger = bigger, smaller
        return smaller, bigger

    @staticmethod
    def _generate_random_label_for_frame(frame_shape, objects):
        """
        Randomly generate labels for one frame.
        :param frame_shape: Dimensions of the frame in format (height, width).
        :param objects: List of tuples containing string and int representing types of objects and their ids
                        for which the labels will be generated. Number of objects in this list corresponds to
                        the number of objects in the label. Objects do not have to be unique.
        :return: The random label for one frame.
        """
        if not len(frame_shape) == 2:
            raise ValueError('Frames have exactly 2 dimensions.')

        frame_label = []
        for object_type, object_id in objects:
            top, bottom = MaskCreationTests._generate_direction_bounds(frame_shape[0])
            left, right = MaskCreationTests._generate_direction_bounds(frame_shape[1])
            frame_label.append((object_type, object_id, top, bottom, left, right))
        return frame_label

    def _test_generated_label_validity(self, frame_shape, objects, label):
        self.assertEqual(len(label), len(objects))
        for i in range(len(objects)):
            object_type, object_id = objects[i]
            object_label = label[i]
            self.assertEqual(len(object_label), 6)
            self.assertEqual(object_type, object_label[0])
            self.assertEqual(object_id, object_label[1])

            top, bottom, left, right = object_label[2:]

            self.assertTrue(left < right)
            self.assertTrue(top < bottom)
            for bound in (left, right):
                self.assertTrue(bound >= 0)
                self.assertTrue(bound < frame_shape[1])
            for bound in (top, bottom):
                self.assertTrue(bound >= 0)
                self.assertTrue(bound < frame_shape[0])

    @staticmethod
    def _generate_random_frame_shape(max_height, max_width):
        if max_height <= 2 or max_width <= 2:
            raise ValueError('Max width and max height must both be bigger than 2.')
        return random.randint(2, max_height), random.randint(2, max_width)

    @staticmethod
    def _generate_objects(count, all_unique=True, random_count=True, _object_id_offset=0):
        if count == 1:
            all_unique = True
        if not all_unique:
            count = int(count / 2)
        object_count = random.randint(1, count) if random_count else count
        objects = [('object_%s' % i, _object_id_offset + i) for i in range(object_count)]
        return objects if all_unique else \
            objects + MaskCreationTests._generate_objects(count, all_unique=True,
                                                          random_count=random_count, _object_id_offset=len(objects))

    @staticmethod
    def _generate_allowed_objects_list(count):
        return ['object_%s' % i for i in range(count)]

    def test_label_generating(self):
        # Test when objects in the frame have unique types
        for i in range(10):
            unique_objects = MaskCreationTests._generate_objects(50)
            unique_objects_frame_shape = MaskCreationTests._generate_random_frame_shape(1000, 1000)
            self._test_generated_label_validity(unique_objects_frame_shape, unique_objects,
                                                MaskCreationTests._generate_random_label_for_frame(
                                                    unique_objects_frame_shape, unique_objects))

        # Test when objects in the frame have non unique types
        for i in range(10):
            none_unique_objects = MaskCreationTests._generate_objects(50, all_unique=False)
            non_unique_objects_frame_shape = MaskCreationTests._generate_random_frame_shape(1000, 1000)
            self._test_generated_label_validity(non_unique_objects_frame_shape, none_unique_objects,
                                                MaskCreationTests._generate_random_label_for_frame(
                                                    non_unique_objects_frame_shape, none_unique_objects))

    TEST_REPETITIONS = 1

    def _test_masks_validity(self, frame_shape, labels, masks, allowed_objects=None):
        mask_shape = masks[0].shape
        vertical_scale = frame_shape[0] / mask_shape[0]
        horizontal_scale = frame_shape[1] / mask_shape[1]
        for i in range(len(labels)):
            frame_labels = labels[i]
            mask = masks[i]
            for i in range(mask_shape[0]):  # Height index
                for j in range(mask_shape[1]):  # Width index
                    mask_value = mask[i, j]
                    # Ranges of pixels in the frame to which given mask pixel maps
                    frame_vertical_coords_range = [round(k * vertical_scale) for k in [i, i + 1]]
                    frame_vertical_coords_range[1] -= 1
                    frame_vertical_coords_range = np.clip(frame_vertical_coords_range, 0, frame_shape[0] - 1)
                    frame_horizontal_coords_range = [round(k * horizontal_scale) for k in [j, j + 1]]
                    frame_horizontal_coords_range[1] -= 1
                    frame_horizontal_coords_range = np.clip(frame_horizontal_coords_range, 0, frame_shape[1] - 1)
                    contained_objects = []
                    for l in frame_vertical_coords_range:
                        for m in frame_horizontal_coords_range:
                            for label in frame_labels:
                                object_type = label[0]
                                if not allowed_objects or object_type in allowed_objects:
                                    top, bottom, left, right = label[2:]
                                    if top <= l <= bottom and left <= m <= right and label not in contained_objects:
                                        contained_objects.append(label)

                    message = 'Invalid mask value on mask coordinates (%s, %s), frame coordinates ranges %s/%s. ' \
                              'Frame shape: %s, mask shape: %s.' \
                              % (i, j, str(frame_vertical_coords_range), str(frame_horizontal_coords_range),
                                 frame_shape, mask_shape)
                    if len(contained_objects) > 0:
                        self.assertEqual(mask_value, 1,
                                         message + ' Should be 1 according to label %s' % str(contained_objects))
                    else:
                        self.assertEqual(mask_value, 0, message + ' Should be 0.')

    def test_single_frame_sequence(self):
        for i in range(MaskCreationTests.TEST_REPETITIONS):
            frame_shape = MaskCreationTests._generate_random_frame_shape(100, 100)
            print('%s, run %s: frame shape %s' % (inspect.stack()[0][3], i, str(frame_shape)))
            frame_label = MaskCreationTests._generate_random_label_for_frame(
                frame_shape,
                MaskCreationTests._generate_objects(30, all_unique=True, random_count=True))
            masks = create_masks([1] + list(frame_shape), [frame_label], frame_shape)
            self._test_masks_validity(frame_shape, [frame_label], masks)

    def test_random_length_frame_sequence_and_half_allowed_objects(self):
        for i in range(MaskCreationTests.TEST_REPETITIONS):
            sequence_frame_count = random.randint(1, 10)
            frame_shape = MaskCreationTests._generate_random_frame_shape(100, 100)
            print('%s, run %s: frame shape %s' % (inspect.stack()[0][3], i, str(frame_shape)))
            frame_labels = [MaskCreationTests._generate_random_label_for_frame(
                frame_shape,
                MaskCreationTests._generate_objects(30, all_unique=False, random_count=True))
                            for i in range(sequence_frame_count)]
            allowed_objects = MaskCreationTests._generate_allowed_objects_list(7)
            masks = create_masks([sequence_frame_count] + list(frame_shape), frame_labels, frame_shape,
                                 allowed_object_types=allowed_objects)
            self._test_masks_validity(frame_shape, frame_labels, masks, allowed_objects=allowed_objects)

    def test_zero_objects_in_frame(self):
        sequence_frame_count = 3
        for i in range(MaskCreationTests.TEST_REPETITIONS):
            frame_shape = MaskCreationTests._generate_random_frame_shape(100, 100)
            print('%s, run %s: frame shape %s' % (inspect.stack()[0][3], i, str(frame_shape)))
            frame_labels = [MaskCreationTests._generate_random_label_for_frame(frame_shape, [])
                            for i in range(sequence_frame_count)]
            masks = create_masks([sequence_frame_count] + list(frame_shape), frame_labels, frame_shape)
            self._test_masks_validity(frame_shape, frame_labels, masks)

    def test_single_object_in_frame(self):
        sequence_frame_count = 3
        for i in range(MaskCreationTests.TEST_REPETITIONS):
            frame_shape = MaskCreationTests._generate_random_frame_shape(100, 100)
            print('%s, run %s: frame shape %s' % (inspect.stack()[0][3], i, str(frame_shape)))
            frame_labels = [MaskCreationTests._generate_random_label_for_frame(
                frame_shape, MaskCreationTests._generate_objects(1, random_count=False))
                            for i in range(sequence_frame_count)]
            masks = create_masks([sequence_frame_count] + list(frame_shape), frame_labels, frame_shape)
            self._test_masks_validity(frame_shape, frame_labels, masks)

    def test_mask_half_the_frame_shape(self):
        for i in range(MaskCreationTests.TEST_REPETITIONS):
            sequence_frame_count = random.randint(1, 10)
            # Make mask shape half the dimensions of frame shape
            mask_shape = MaskCreationTests._generate_random_frame_shape(50, 50)
            frame_shape = [i * 2 for i in mask_shape]
            print('%s, run %s: frame shape %s' % (inspect.stack()[0][3], i, str(frame_shape)))
            frame_labels = [MaskCreationTests._generate_random_label_for_frame(
                frame_shape,
                MaskCreationTests._generate_objects(1, all_unique=False, random_count=True))  # 20
                            for i in range(sequence_frame_count)]
            masks = create_masks([sequence_frame_count] + list(frame_shape), frame_labels, mask_shape)
            self._test_masks_validity(frame_shape, frame_labels, masks)


class TestDataset():
    def __init__(self, sample_count, validation_split):
        self.sample_count = sample_count
        self.validation_split = validation_split

        self.num_sets = round(1 / validation_split)
        self.validation_samples_count = round(100 * validation_split)
        self.training_samples_count = self.sample_count - self.validation_samples_count

        self.x = np.array(['sample_%s' % i for i in range(self.sample_count)])
        self.y = np.array(['label_%s' % i for i in range(self.sample_count)])

    def __str__(self):
        return type(self).__name__ \
               + ' sample_count: %d, validation_split: %s' % (self.sample_count, self.validation_split)


class DataSplittingTests(unittest.TestCase):
    def _test_data_split(self, data_set, splits, split_index):
        message = 'Dataset: (%s), splits_count: %s, split_index: %s' % (str(data_set), len(splits), split_index)

        split = splits[split_index]
        x_train, y_train, x_val, y_val = split[0], split[1], split[2], split[3]

        # Check that generated sets have proper sizes
        self.assertEqual(len(x_train), data_set.training_samples_count, message)
        self.assertEqual(len(y_train), data_set.training_samples_count, message)
        self.assertEqual(len(x_val), data_set.validation_samples_count, message)
        self.assertEqual(len(y_val), data_set.validation_samples_count, message)

        # Check that samples are properly aligned
        for data_part in [(x_train, y_train), (x_val, y_val)]:
            for j in range(len(data_part[0])):
                sample = data_part[0][j]
                label = data_part[1][j]
                sample_num = sample.split('_')[1]
                self.assertEqual(sample, 'sample_%s' % sample_num, message)
                self.assertEqual(label, 'label_%s' % sample_num, message)

        # Check that no sample in the validation set is in the training set
        for validation_sample in x_val:
            self.assertTrue(validation_sample not in x_train, message)

        # Check that no sample in the validation set is also in any other validation set of other splits
        for j in list(range(split_index)) + list(range(split_index + 1, len(splits))):
            other_x_val = splits[j][2]
            for val_sample in x_val:
                self.assertTrue(val_sample not in other_x_val,
                                '%s: Sample %s from validation set on index %d was found in '
                                'validation set on index %d' % (message, val_sample, split_index, j))

    def test_all_sets_requested(self):
        for data_set in [TestDataset(100, 0.2), TestDataset(101, 0.2)]:
            splits = split_data(data_set.x, data_set.y, data_set.validation_split)
            self.assertEqual(len(splits), data_set.num_sets)
            for i in range(data_set.num_sets):
                self._test_data_split(data_set, splits, i)

    def test_not_all_sets_requested(self):
        for data_set in [TestDataset(100, 0.2), TestDataset(101, 0.2)]:
            for num_sets in range(1, data_set.num_sets + 1):
                splits = split_data(data_set.x, data_set.y, data_set.validation_split, num_sets=num_sets)
                self.assertEqual(len(splits), num_sets)
                for i in range(num_sets):
                    self._test_data_split(data_set, splits, i)


if __name__ == '__main__':
    unittest.main()
