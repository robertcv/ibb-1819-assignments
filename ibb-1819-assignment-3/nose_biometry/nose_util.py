from copy import copy
from random import shuffle

import cv2 as cv


class Image:
    def __init__(self, location):
        self.location = location
        self.img = cv.cvtColor(cv.imread(location), cv.COLOR_BGR2GRAY)
        self.file = location.split("/")[-1]
        self.subject = location.split("/")[-2]

    def crop(self, x, y, w, h):
        self.img = self.img[y:y+h, x:x+w]

    def resize(self, w, h):
        self.img = cv.resize(self.img, (w, h))

    def __repr__(self):
        return self.subject + " - " + self.file

    def __copy__(self):
        return Image(self.location)


class ImageDataBase(list):
    def __init__(self, info_file):
        super().__init__()
        self.info_file = info_file
        prefix = info_file.split("/")[:-1]
        with open(info_file) as f:
            self.image_locations = ["/".join(prefix) + "/" + l.split()[0]
                                    for l in f if l.split()[1] != '-1']

    def read_images(self, n=-1):
        if n == -1:
            self[:] = [Image(i) for i in self.image_locations]
        else:
            self[:] = [Image(i) for i in self.image_locations[:n]]

    def shuffle(self):
        shuffle(self)

    def test_set(self):
        return [img for img in self if img.subject.startswith("robertcv")]

    def new_data(self, new_images: list):
        self[:] = new_images

    def __copy__(self):
        new_db = ImageDataBase(self.info_file)
        new_db.new_data([copy(i) for i in self])
        return new_db


if __name__ == '__main__':
    images = ImageDataBase("nose_biometry/data/info.csv")
    images.read_images(10)

    print(images)


