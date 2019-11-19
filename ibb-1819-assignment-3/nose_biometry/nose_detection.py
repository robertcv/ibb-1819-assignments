import cv2 as cv

from nose_biometry.nose_util import Image


class Detector:
    """Base class to implement the nose detector."""
    def detect(self, image: Image, **kwargs):
        """
        Detect the nose and return the rectangle
        :param image: Image object
        :return: (x1, y1, x2, y2)
        """
        raise NotImplementedError


class TrueDetector(Detector):

    def __init__(self, info_file):
        with open(info_file) as f:
            self.info = [l.strip().split() for l in f]

    def detect(self, image, **kwargs):
        try:
            _, x, y, w, h = list(filter(lambda x: x[0] in image.location,
                                        self.info))[0]
            return int(x), int(y), int(w), int(h)
        except:
            return


class OpenCVCascadeDetector(Detector):

    def __init__(self, cascade_file):
        self.cascade = cv.CascadeClassifier(cascade_file)

    def detect(self, image, **kwargs):
        noses = self.cascade.detectMultiScale3(image.img, kwargs["scaleFactor"],
                                               kwargs["minNeighbors"],
                                               outputRejectLevels=True)
        if noses[0] == ():
            return
        max_nose = max(zip(*noses), key=lambda x: x[2])
        x, y, w, h = max_nose[0]
        return x, y, w, h


class IntersectionOverUnion:
    """Calculate the intersection over union score"""
    def __init__(self, images):
        self.images = images
        self.iou = None

    def calc_avg(self, true_detector, predict_detector, **kwargs):
        if self.iou is None:
            self.iou = [self._calc_iou(image, true_detector, predict_detector, kwargs)
                        for image in self.images]
        return sum(self.iou) / len(self.iou)

    def calc_not_miss(self, true_detector, predict_detector, **kwargs):
        if self.iou is None:
            self.iou = [self._calc_iou(image, true_detector, predict_detector, kwargs)
                        for image in self.images]
        return len(list(filter(lambda x: x > 0.5, self.iou))) / len(self.iou)

    @staticmethod
    def _calc_iou(img, td, pd, kwargs):
        x, y, w, h = td.detect(img)

        detection = pd.detect(img, **kwargs)
        if detection is None:
            return 0

        dx, dy, dw, dh = detection
        xA = max(x, dx)
        yA = max(y, dy)
        xB = min(x + w, dx + dw)
        yB = min(y + h, dy + dh)

        intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxA = (x + w - x + 1) * (y + h - y + 1)
        boxB = (dx + dw - dx + 1) * (dy + dh - dy + 1)

        return intersection / (boxA + boxB - intersection)


if __name__ == '__main__':
    from nose_biometry.nose_util import ImageDataBase
    images = ImageDataBase("nose_biometry/data/info.csv")
    images.read_images(10)

    image = images[1]

    td = TrueDetector("nose_biometry/data/info.csv")
    # x, y, w, h = td.detect(image)
    # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cvd = OpenCVCascadeDetector('nose_biometry/data/haarcascade_mcs_nose.xml')
    # x, y, w, h = cvd.detect(image, scaleFactor=1.3, minNeighbors=5)
    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    mycvd = OpenCVCascadeDetector('nose_biometry/data/haarcascade_mcs_nose.xml')
    # x, y, w, h = mycvd.detect(image, scaleFactor=1.1, minNeighbors=1)
    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # cv.imshow("img", image.img)
    # cv.waitKey(0)

    iou = IntersectionOverUnion(images)
    # print(iou.calc_avg(td, cvd, scaleFactor=1.3, minNeighbors=5)) --> 0.434
    print(iou.calc_not_miss(td, cvd, scaleFactor=1.3, minNeighbors=5))
    iou = IntersectionOverUnion(images)
    # print(iou.calc_avg(td, mycvd, scaleFactor=1.3, minNeighbors=5)) --> 0.292
    print(iou.calc_not_miss(td, mycvd, scaleFactor=1.3, minNeighbors=5))
