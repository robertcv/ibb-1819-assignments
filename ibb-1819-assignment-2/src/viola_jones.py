import sys
import numpy as np
import cv2 as cv

from src.file_util import *


# left_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_leftear.xml')
# right_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_rightear.xml')

left_ear_cascade = cv.CascadeClassifier('data/opencv/cascade_left_ear_equal.xml')
right_ear_cascade = cv.CascadeClassifier('data/opencv/cascade_right_ear_equal.xml')


def calc_ear_mask(images, scaleFactor, minNeighbors):
    result = []
    for image in images:
        print("Detect:", image)
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        left_ears = left_ear_cascade.detectMultiScale3(gray,
                                                       scaleFactor,
                                                       minNeighbors,
                                                       outputRejectLevels=True)
        right_ears = right_ear_cascade.detectMultiScale3(gray,
                                                         scaleFactor,
                                                         minNeighbors,
                                                         outputRejectLevels=True)

        if right_ears[0] != () and left_ears[0] != ():
            combine_max = max([
                max(zip(*left_ears), key=lambda x: x[2]),
                max(zip(*right_ears), key=lambda x: x[2])],
                key=lambda x: x[2])
        elif right_ears[0] == () and left_ears[0] != ():
            combine_max = max(zip(*left_ears), key=lambda x: x[2])
        elif right_ears[0] != () and left_ears[0] == ():
            combine_max = max(zip(*right_ears), key=lambda x: x[2])
        else:
            result.append(mask)
            continue

        x, y, w, h = combine_max[0]
        cv.rectangle(mask, (x, y), (x + w, y + h), 255, cv.FILLED)
        result.append(mask)

    return result


def get_ear_mask(images):
    return [cv.imread(image, 0) for image in images]


def intersection_over_union(calc, true):
    return np.sum(calc & true) / np.sum(calc | true)


if __name__ == '__main__':
    _, _, x_test, y_test = learn_test("andrazpov-info.csv", "robertcv-info.csv")
    if len(sys.argv) == 3:
        scaleFactor = int(sys.argv[1])
        minNeighbors = int(sys.argv[2])
    else:
        scaleFactor = 1.1
        minNeighbors = 11
    print("Start detecting")
    y_calc = calc_ear_mask(x_test, scaleFactor, minNeighbors)
    y_true = get_ear_mask(y_test)
    print("Start calculating score")
    scores = np.array([intersection_over_union(calc, true)
                       for calc, true in zip(y_calc, y_true)])

    print("par: {} {}".format(scaleFactor, minNeighbors))
    print(len(scores), np.sum(scores > 0.5))
    print(np.mean(scores))
