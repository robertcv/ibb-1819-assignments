from nose_biometry.nose_util import ImageDataBase
from nose_biometry.nose_detection import TrueDetector
from nose_biometry.nose_extraction import HOGExtractor, LBPExtractor
from nose_biometry.nose_models import KNNModel
from nose_biometry.nose_pipline import BiometryPipeline
from nose_biometry.nose_visualize import plot_cmc

images = ImageDataBase("data/info.csv")
images.read_images()

pipeline1 = BiometryPipeline("hog-knn", images)
pipeline1.add_detector(TrueDetector("data/info.csv"))
pipeline1.add_extractor(HOGExtractor(), block_norm='L2-Hys')
pipeline1.add_model(KNNModel(n_neighbors=15))

pipeline1.cross_validation()
print(pipeline1.calc_score("accuracy"))

pipeline3 = BiometryPipeline("lbp-knn", images)
pipeline3.add_detector(TrueDetector("data/info.csv"))
pipeline3.add_extractor(LBPExtractor(), numPoints=48, radius=16)
pipeline3.add_model(KNNModel(n_neighbors=15))

plot_cmc([pipeline1, pipeline3], save="cmc.png")
