from copy import copy

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import clone
from scipy import interp

from nose_biometry.nose_detection import Detector
from nose_biometry.nose_extraction import Extractor
from nose_biometry.nose_models import Model
from nose_biometry.nose_util import Image, ImageDataBase


class BiometryPipeline:
    def __init__(self, label, images_db: ImageDataBase, resize=(100, 100),
                 verbose=True):
        self.label = label
        self.images_db = copy(images_db)
        self.resize = resize
        self.verbose = verbose
        self.detector = None
        self.detector_kwargs = None
        self.extractor = None
        self.extractor_kwargs = None
        self.features = None
        self.classes = None
        self.classes_predict = None
        self.classes_predict_proba = None
        self.model = None
        self.model_kwargs = None

    def add_detector(self, detector: Detector, **kwargs):
        self.detector = detector
        self.detector_kwargs = kwargs

    def add_extractor(self, extractor: Extractor, **kwargs):
        self.extractor = extractor
        self.extractor_kwargs = kwargs

    def add_model(self, model: Model):
        self.model = model

    def _print_verbose(self, msg):
        if self.verbose:
            print(msg)

    def _run_detection(self):
        self._print_verbose("Start detecting!")
        new_images_db = []
        for image in self.images_db:
            new_image = self._run_detection_on_image(image)
            if new_image is not None:
                new_images_db.append(new_image)
        if len(new_images_db) != len(self.images_db):
            print("WARNING: Only {} out of {} noses were detected!".format(
                len(new_images_db), len(self.images_db)))
        self.images_db.new_data(new_images_db)

    def _run_detection_on_image(self, image: Image):
        box = self.detector.detect(image, **self.detector_kwargs)
        if box is None:
            return
        image.crop(*box)
        image.resize(*self.resize)
        return image

    def _run_extraction(self):
        self._print_verbose("Start feature extraction!")
        features = [self._run_extraction_on_image(image)
                    for image in self.images_db]
        classes = [image.subject for image in self.images_db]
        self.features = np.array(features)
        self.classes = np.array(classes)

    def _run_extraction_on_image(self, image: Image):
        return self.extractor.extract(image, **self.extractor_kwargs)

    def cross_validation(self, folds=5):
        if self.features is None:
            self.images_db.shuffle()
            self._run_detection()
            self._run_extraction()
        self._print_verbose("Start cross-validation!")
        self.classes_predict = self.classes.copy()
        self.classes_predict_proba = np.zeros((len(self.classes),
                                               len(np.unique(self.classes))))
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(self.features):
            model_clone = clone(self.model)
            model_clone.fit(self.features[train_index],
                           self.classes[train_index])
            self.classes_predict[test_index] = \
                model_clone.predict(self.features[test_index])
            self.classes_predict_proba[test_index] = \
                model_clone.predict_proba(self.features[test_index])

    def calc_score(self, score):
        if score == 'accuracy':
            return accuracy_score(self.classes, self.classes_predict)
        elif score == 'f1':
            return f1_score(self.classes, self.classes_predict,
                            average='weighted')
        elif score == 'precision':
            return precision_score(self.classes, self.classes_predict,
                                   average='weighted')
        elif score == 'recall':
            return recall_score(self.classes, self.classes_predict,
                                average='weighted')
        elif score == 'roc_auc':
            return roc_auc_score(self.classes, self.classes_predict,
                                 average='weighted')

    def calc_roc(self):
        if self.features is None:
            self.images_db.shuffle()
            self._run_detection()
            self._run_extraction()
        self._print_verbose("Calculate ROC")

        sorted_unique_classes = np.sort(np.unique(self.classes))
        n_classes = len(sorted_unique_classes)
        y = label_binarize(self.classes, classes=sorted_unique_classes)

        y_score = np.zeros((len(y), n_classes))
        classifier = OneVsRestClassifier(self.model)

        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(self.features):
            classifier_clone = clone(classifier)
            y_score[test_index] = classifier_clone\
                .fit(self.features[train_index], y[train_index])\
                .predict_proba(self.features[test_index])

        fpr = dict()
        tpr = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])

        dense_fpr = np.linspace(0, 1, num=101)
        dense_tpr = np.array([interp(dense_fpr, fpr[i], tpr[i])
                             for i in range(n_classes)])
        mean_tpr = np.mean(dense_tpr, axis=0)
        std_tpr = np.std(dense_tpr, axis=0)
        mean_auc = auc(dense_fpr, mean_tpr)
        return dense_fpr, mean_tpr, std_tpr, mean_auc

    def calc_cmc(self):
        if self.features is None:
            self.images_db.shuffle()
            self._run_detection()
            self._run_extraction()
        self._print_verbose("Calculate CMC")

        sorted_unique_classes = np.sort(np.unique(self.classes))
        y_proba = np.zeros((len(self.classes), len(sorted_unique_classes)))
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(self.features):
            model_clone = clone(self.model)
            model_clone.fit(self.features[train_index],
                            self.classes[train_index])
            y_proba[test_index] = model_clone.predict_proba(
                self.features[test_index])

        y_rank = np.zeros(len(self.classes))
        y_sort = np.argsort(-y_proba)
        for i in range(len(self.classes)):
            j = np.where(sorted_unique_classes == self.classes[i])[0][0]
            y_rank[i] = np.where(y_sort[i] == j)[0][0] + 1

        unique, counts = np.unique(y_rank, return_counts=True)
        counts = counts / np.sum(counts)
        return unique, np.cumsum(counts)

    def fit(self):
        if self.features is None:
            self.images_db.shuffle()
            self._run_detection()
            self._run_extraction()
        self.model.fit(self.features, self.classes)

    def predict(self, image):
        new_image = self._run_detection_on_image(image)
        if new_image is None:
            print("WARNING: No detection found!")
            return
        features = self._run_extraction_on_image(new_image)
        return self.model.predict(features)


if __name__ == '__main__':
    from nose_biometry.nose_detection import TrueDetector
    from nose_biometry.nose_extraction import HOGExtractor
    from nose_biometry.nose_models import SVMModel

    images = ImageDataBase("data/info.csv")
    images.read_images(100)

    pipeline = BiometryPipeline("Test pipeline", images)
    pipeline.add_detector(TrueDetector("data/info.csv"))
    pipeline.add_extractor(HOGExtractor(), block_norm='L2-Hys')
    pipeline.add_model(SVMModel(gamma='auto'))
    pipeline.cross_validation()
    print(pipeline.calc_score('accuracy'))
