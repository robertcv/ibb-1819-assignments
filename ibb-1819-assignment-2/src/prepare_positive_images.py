import os
import os.path as path
import cv2 as cv
from src.file_util import good_left_images, good_right_images

pos_path = path.join("data", "opencv")


def relocate_pos(images, location):
    location = os.path.join(pos_path, location)
    if not os.path.exists(location):
        os.makedirs(location)

    n = 1
    for image in images:
        img = cv.imread(path.join("data", "ears-cropped", image))
        new_location = path.join(location, str(n) + ".jpg")
        img = cv.resize(img, (100, 100))
        cv.imwrite(new_location, img)
        n += 1


if __name__ == '__main__':
    relocate_pos(good_left_images("andrazpov-info.csv"), "pos_left")
    relocate_pos(good_right_images("andrazpov-info.csv"), "pos_right")
