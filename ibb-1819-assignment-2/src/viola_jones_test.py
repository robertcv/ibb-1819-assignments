import cv2 as cv
import numpy as np


def find(img, left, right, s, n):
    left_ears = left.detectMultiScale3(img, s, n, outputRejectLevels=True)
    right_ears = right.detectMultiScale3(img, s, n, outputRejectLevels=True)

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
        return 0, 0, 0, 0

    return combine_max[0]


def from_mask(mask):
    index = np.argwhere(np.diff(np.diff(mask, axis=0), axis=1) == 255)
    return index[0, 1], index[0, 0], \
           index[1, 1] - index[0, 1], index[1, 0] - index[0, 0]


if __name__ == '__main__':
    # F: 01/16, 01/18, 01/19
    # T: 02/22
    i = "robertcv-16/13.png"
    img = cv.imread("data/images/" + i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = cv.imread("data/masks-rectangular/" + i,
                     cv.IMREAD_GRAYSCALE)

    x, y, w, h = find(
        gray,
        cv.CascadeClassifier('haarcascade_mcs_leftear.xml'),
        cv.CascadeClassifier('haarcascade_mcs_rightear.xml'),
        1.05, 1
    )
    #cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    x, y, w, h = find(
        gray,
        cv.CascadeClassifier('data/opencv/cascade_left_ear_equal.xml'),
        cv.CascadeClassifier('data/opencv/cascade_right_ear_equal.xml'),
        1.1, 11
    )
    #cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    x, y, w, h = from_mask(mask)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv.imwrite('bead_partioal2.png', img)
    #cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
