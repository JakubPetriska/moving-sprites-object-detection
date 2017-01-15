import random
import unittest

import numpy as np

from localization.training_utils import split_data


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
            frame_label.append((object_type, object_id, left, top, right, bottom))
        return frame_label

    def _test_generated_label_validity(self, frame_shape, objects, label):
        self.assertEqual(len(label), len(objects))
        for i in range(len(objects)):
            object_type, object_id = objects[i]
            object_label = label[i]
            self.assertEqual(len(object_label), 6)
            self.assertEqual(object_type, object_label[0])
            self.assertEqual(object_id, object_label[1])

            left = object_label[2]
            top = object_label[3]
            right = object_label[4]
            bottom = object_label[5]

            self.assertTrue(left < right)
            self.assertTrue(top < bottom)
            for bound in (left, right):
                self.assertTrue(bound >= 0)
                self.assertTrue(bound < frame_shape[1])
            for bound in (top, bottom):
                self.assertTrue(bound >= 0)
                self.assertTrue(bound < frame_shape[0])

    def test_label_generating(self):
        for i in range(10):
            non_duplicit_objects = [('object_%s' % i, i) for i in range(5000)]
            non_duplicit_objects_frame_shape = (random.randint(1, 1000), random.randint(1, 1000))
            self._test_generated_label_validity(non_duplicit_objects_frame_shape, non_duplicit_objects,
                                                self._generate_random_label_for_frame(non_duplicit_objects_frame_shape,
                                                                                      non_duplicit_objects))

        for i in range(10):
            duplicit_objects = [('object_%s' % i, i) for i in range(5000)] + [('object_%s' % i, i) for i in range(1000)]
            duplicit_objects_frame_shape = (random.randint(1, 1000), random.randint(1, 1000))
            self._test_generated_label_validity(duplicit_objects_frame_shape, duplicit_objects,
                                                self._generate_random_label_for_frame(duplicit_objects_frame_shape,
                                                                                      duplicit_objects))

    def test_mask_creation(self):
        # TODO
        pass


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
