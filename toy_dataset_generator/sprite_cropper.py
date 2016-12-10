# Crop all sprites of any overhanging background.

from os import listdir
import os.path
from scipy import misc
import numpy as np

from toy_dataset_generator import constants


file_names = listdir(constants.SPRITES_DIR)
index = 0
for file_name in file_names:
    file_path = os.path.join(constants.SPRITES_DIR, file_name)
    sprite = misc.imread(file_path)
    transparent_pixel = np.zeros(sprite.shape[2])

    # Cut out top empty pixels
    for i in range(0, sprite.shape[0]):
        should_break = False
        for j in range(0, sprite.shape[1]):
            if not sprite[i, j, 3] == 0:
                sprite = sprite[i:, :]
                should_break = True
                break
        if should_break:
            break

    # Cut out empty bottom pixels
    for i in range(sprite.shape[0] - 1, -1, -1):
        should_break = False
        for j in range(0, sprite.shape[1]):
            if not sprite[i, j, 3] == 0:
                sprite = sprite[0:i, :]
                should_break = True
                break
        if should_break:
            break

    # Cut out empty left pixels
    for j in range(0, sprite.shape[1]):
        should_break = False
        for i in range(0, sprite.shape[0]):
            if not sprite[i, j, 3] == 0:
                sprite = sprite[:, j:]
                should_break = True
                break
        if should_break:
            break

    # Cut out empty right pixels
    for j in range(sprite.shape[1] - 1, -1, -1):
        should_break = False
        for i in range(0, sprite.shape[0]):
            if not sprite[i, j, 3] == 0:
                sprite = sprite[:, 0:j]
                should_break = True
                break
        if should_break:
            break

    misc.imsave(file_path, sprite)
    index += 1
    print('%s/%s' % (index, len(file_names)))
