import unittest

import numpy as np

from localization.training_utils import split_data


class MaskCreationTests(unittest.TestCase):
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
