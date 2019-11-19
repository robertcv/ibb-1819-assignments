import matplotlib.pyplot as plt


def plot_roc(pipelines, save=None):
    if not isinstance(pipelines, list):
        pipelines = [pipelines]

    fig, ax = plt.subplots(1)

    for pl in pipelines:
        x, y, _, auc = pl.calc_roc()
        ax.plot(x, y, label="{} (area={:0.2f})".format(pl.label, auc))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    ax.legend(loc="lower right")
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def plot_cmc(pipelines, save=None):
    if not isinstance(pipelines, list):
        pipelines = [pipelines]

    fig, ax = plt.subplots(1)

    for pl in pipelines:
        x, y, = pl.calc_cmc()
        ax.plot(x, y, label="{}".format(pl.label))

    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('Rank')
    ax.set_ylabel('Rate')
    ax.set_title('CMC curve')
    ax.legend(loc="lower right")
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


if __name__ == '__main__':
    from nose_biometry.nose_util import ImageDataBase
    from nose_biometry.nose_pipline import BiometryPipeline
    from nose_biometry.nose_detection import TrueDetector
    from nose_biometry.nose_extraction import LBPExtractor, HOGExtractor
    from nose_biometry.nose_models import KNNModel

    images = ImageDataBase("data/info.csv")
    images.read_images(500)

    pipeline1 = BiometryPipeline("hog", images)
    pipeline1.add_detector(TrueDetector("data/info.csv"))
    pipeline1.add_extractor(HOGExtractor())
    pipeline1.add_model(KNNModel())

    pipeline2 = BiometryPipeline("lbp", images)
    pipeline2.add_detector(TrueDetector("data/info.csv"))
    pipeline2.add_extractor(LBPExtractor(), numPoints=48, radius=16)
    pipeline2.add_model(KNNModel())

    # plot_roc([pipeline1, pipeline2])
    plot_cmc([pipeline1, pipeline2])
