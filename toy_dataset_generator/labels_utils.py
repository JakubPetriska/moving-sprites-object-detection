import csv

LABELS_FILE_DELIMITER = ';'


def write_labels(output_path, labels):
    with open(output_path, 'w', newline='') as labelsfile:
        labels_writer = csv.writer(labelsfile, delimiter=LABELS_FILE_DELIMITER)
        for frame_labels in labels:
            labels_writer.writerow(frame_labels)


def read_labels(input_path):
    # with open(output_path, newline='') as labelsfile:
    #     labelsreader = csv.reader(labelsfile, delimiter=LABELS_FILE_DELIMITER, quotechar='|
    pass
