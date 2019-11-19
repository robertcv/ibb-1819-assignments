from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class Model:
    def __init__(self, model, **kwargs):
        self.model = model(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)


class KNNModel(Model):
    def __init__(self, **kwargs):
        super().__init__(KNeighborsClassifier, **kwargs)


class SVMModel(Model):
    def __init__(self, **kwargs):
        kwargs.pop('probability', None)
        super().__init__(SVC, probability=True, **kwargs)


if __name__ == '__main__':
    from nose_biometry.nose_util import ImageDataBase
    from nose_biometry.nose_extraction import LBPExtractor, HOGExtractor
    from sklearn.model_selection import cross_val_score
    import numpy as np

    images = ImageDataBase("data/info.csv")
    print("Loading data")
    images.read_images(50)
    images.shuffle()
    print("Extracting features")
    lbp = LBPExtractor()
    hog = HOGExtractor()
    X = []
    y = []
    for image in images:
        image.resize(100, 100)
        # X.append(lbp.extract(image, numPoints=48, radius=16))
        X.append(hog.extract(image))
        y.append(image.subject)
    print("Cross-validating")
    knn = KNNModel()
    print(cross_val_score(knn, np.array(X), np.array(y), scoring='accuracy', cv=5))
    knn = SVMModel(gamma='auto')
    print(cross_val_score(knn, np.array(X), np.array(y), scoring='accuracy', cv=5))
