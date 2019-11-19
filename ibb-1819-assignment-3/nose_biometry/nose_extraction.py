from skimage import feature
import numpy as np

from nose_biometry.nose_util import Image


class Extractor:
    """Base class to implement nose feature extractor."""
    def extract(self, image: Image, **kwargs):
        """
        Returne extracted features from image
        :param image: Image object
        :return: feature vector
        """
        raise NotImplementedError


class LBPExtractor(Extractor):

    def extract(self, image, **kwargs):

        lbp = feature.local_binary_pattern(image.img, kwargs["numPoints"],
                                           kwargs["radius"], method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, kwargs["numPoints"] + 3),
                                 range=(0, kwargs["radius"] + 2))

        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)

        return hist


class HOGExtractor(Extractor):

    def extract(self, image, **kwargs):

        f = feature.hog(image.img, **kwargs, feature_vector=True)

        return f


if __name__ == '__main__':
    from nose_biometry.nose_util import ImageDataBase
    images = ImageDataBase("nose_biometry/data/info.csv")
    images.read_images(10)

    lbp = LBPExtractor()
    hog = HOGExtractor()
    for image in images:
        print(lbp.extract(image, numPoints=64, radius=8))
        print(hog.extract(image))
