import os
import sys

import numpy as np
from scipy import misc
from scipy.cluster.vq import kmeans

from detection import constants
from detection.model import Model
from detection.utils import annotate_image
from detection.utils import save_mask
from detection.utils import save_mask_colored

FILTER_THRESHOLD = 0.65
CENTROID_DISTANCE_IMPROVEMENT_PERCENTAGE_THRESHOLD = 0.58

SAVE_MASKS = False
SAVE_CLUSTERED_MASKS = False


def find_cluster_assignment(centroids, data):
    data_points_count = len(data)
    centroids_count = len(centroids)
    cluster_assignment = np.zeros(data_points_count, np.int)
    for data_point_index in range(data_points_count):
        data_point = data[data_point_index]
        min_centroid_distance = -1
        closest_centroid_index = -1
        for centroid_index in range(centroids_count):
            centroid_distance = np.sum(np.square(np.subtract(centroids[centroid_index], data_point)))
            if closest_centroid_index == -1 or centroid_distance < min_centroid_distance:
                min_centroid_distance = centroid_distance
                closest_centroid_index = centroid_index
        cluster_assignment[data_point_index] = closest_centroid_index
    return cluster_assignment


def calculate_average_squared_centroid_distance(centroids, data, cluster_assignment):
    squared_centroid_distance_sum = 0
    data_points_count = len(data)
    for data_point_index in range(data_points_count):
        data_point = data[data_point_index]
        data_point_centroid = centroids[cluster_assignment[data_point_index]]
        squared_centroid_distance_sum += np.sum(np.square(np.subtract(data_point_centroid, data_point)))
    return squared_centroid_distance_sum / data_points_count


def find_clusters(data):
    """Do kmeans, starting from k=1 and raise it by 1 after each iteration until average squared
    distance of data points to their centroids stops falling so much.
    """
    cluster_assignment = None
    clusters_count = 0
    average_centroid_distance = -1
    centroid_distance_improvement_percentage = -1
    while average_centroid_distance == -1 \
            or centroid_distance_improvement_percentage > CENTROID_DISTANCE_IMPROVEMENT_PERCENTAGE_THRESHOLD:
        clusters_count += 1
        centroids, distortion = kmeans(data.astype(np.float), clusters_count)
        cluster_assignment = find_cluster_assignment(centroids, data)
        prev_average_centroid_distance = average_centroid_distance
        average_centroid_distance = \
            calculate_average_squared_centroid_distance(centroids, data, cluster_assignment)
        if prev_average_centroid_distance == -1:
            centroid_distance_improvement_percentage = 1
        else:
            centroid_distance_improvement_percentage = \
                (prev_average_centroid_distance - average_centroid_distance) / prev_average_centroid_distance
    return clusters_count, cluster_assignment


def find_cluster_bounds(data, cluster_count, cluster_assignment):
    cluster_bounds = [[-1, -1, -1, -1] for i in range(cluster_count)]
    for data_point_index in range(len(data)):
        data_point = data[data_point_index]
        cluster = cluster_bounds[cluster_assignment[data_point_index]]
        # Update top cluster bound
        if cluster[0] == -1 or data_point[0] < cluster[0]:
            cluster[0] = data_point[0]
        # Update bottom cluster bound
        if cluster[1] == -1 or data_point[0] > cluster[1]:
            cluster[1] = data_point[0]
        # Update left cluster bound
        if cluster[2] == -1 or data_point[1] < cluster[2]:
            cluster[2] = data_point[1]
        # Update bottom cluster bound
        if cluster[3] == -1 or data_point[1] > cluster[3]:
            cluster[3] = data_point[1]
    return cluster_bounds


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python detect.py <model-dir> <output-dir> <input-files>")
        sys.exit(1)

    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_files = sys.argv[3:]
    x = np.empty((len(input_files),
                  constants.RESOLUTION_HEIGHT,
                  constants.RESOLUTION_WIDTH, 3))
    for i, input_file_path in enumerate(input_files):
        x[i] = misc.imread(input_file_path)

    model_dir = sys.argv[1]
    model_file_path = os.path.join(model_dir, 'model.json')
    model_wights_file_path = \
        os.path.join(model_dir, 'model_best_weights.h5')
    model = Model(model_file_path, model_wights_file_path)
    model_output_shape = model.model.layers[-1].output_shape[1:]

    ys = model.predict(x)
    for i, input_file_path in enumerate(input_files):
        input_file_name = ''.join(os.path.split(input_file_path)[1].split('.')[0:-1])
        output_file_path = os.path.join(output_dir, input_file_name)

        y = ys[i]
        y[y < FILTER_THRESHOLD] = 0

        non_zero_coords = np.nonzero(y)
        non_zero_coords = np.array([non_zero_coords[0], non_zero_coords[1]]).T

        cluster_count = 0
        if len(non_zero_coords) > 0:
            cluster_count, cluster_assignment = find_clusters(non_zero_coords)
            cluster_bounds = find_cluster_bounds(non_zero_coords, cluster_count, cluster_assignment)
            scale = np.divide(np.array([constants.RESOLUTION_HEIGHT, constants.RESOLUTION_WIDTH]),
                              np.array(model_output_shape[0:-1]))
            for cluster_bound in cluster_bounds:
                cluster_bound[0] = int(round(cluster_bound[0] * scale[0]))
                cluster_bound[1] = int(round(cluster_bound[1] * scale[0]))
                cluster_bound[2] = int(round(cluster_bound[2] * scale[1]))
                cluster_bound[3] = int(round(cluster_bound[3] * scale[1]))

            annotate_image(x[i], cluster_bounds)

        misc.imsave(output_file_path + '.png', x[i])

        if SAVE_MASKS:
            save_mask(output_file_path + '-mask.png', y)

        if SAVE_CLUSTERED_MASKS:
            y_clustered = np.zeros((44, 44, 3))
            colors = np.random.rand(cluster_count, 3)
            for k, pixel_coords in enumerate(non_zero_coords):
                y_clustered[pixel_coords[0]][pixel_coords[1]] = colors[cluster_assignment[k]]
            save_mask_colored(output_file_path + '-mask-clustered.png', y_clustered)
