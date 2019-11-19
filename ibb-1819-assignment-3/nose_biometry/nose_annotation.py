import os
import sys

from PyQt5.QtWidgets import QLabel, QRubberBand, QWidget, \
    QPushButton, QVBoxLayout, QApplication, QShortcut
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import QRect, QSize, Qt


class QImageEdit(QLabel):
    def __init__(self, parentQWidget = None):
        super(QImageEdit, self).__init__(parentQWidget)
        self.rubberBand = None
        self.move_rubberBand = False
        self.rubberBand_offset = None
        self.originPoint = None

    def setImage(self, image: QPixmap):
        self.setPixmap(image)

    def getImage(self):
        if self.rubberBand is not None:
            return self.rubberBand.geometry()

    def clear(self):
        super(QImageEdit, self).clear()
        if self.rubberBand is not None:
            self.rubberBand.deleteLater()
        self.rubberBand = None
        self.move_rubberBand = False
        self.rubberBand_offset = None
        self.originPoint = None

    def mousePressEvent(self, event):
        self.originPoint = event.pos()

        if self.rubberBand is None:
            self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
            self.rubberBand.setGeometry(QRect(self.originPoint, QSize()))
            self.rubberBand.show()
        else:
            if self.rubberBand.geometry().contains(self.originPoint):
                self.rubberBand_offset = \
                    self.originPoint - self.rubberBand.pos()
                self.move_rubberBand = True
            else:
                self.rubberBand.hide()
                self.rubberBand.deleteLater()
                self.rubberBand = None
                self.move_rubberBand = False
                self.rubberBand_offset = None
                self.mousePressEvent(event)

    def mouseMoveEvent (self, event):
        newPoint = event.pos()
        if self.move_rubberBand:
            self.rubberBand.move(newPoint - self.rubberBand_offset)
        else:
            self.rubberBand.setGeometry(
                QRect(self.originPoint, newPoint).normalized())

    def mouseReleaseEvent (self, event):
        self.move_rubberBand = False


class QMain(QWidget):
    def __init__(self, *args, **kwargs):
        super(QMain, self).__init__(*args, **kwargs)
        self.info_file = os.path.join('data', 'info.csv')
        self.to_observe = []
        self.current_image = None
        self.initData()
        self.next_image_gen = self.get_next_image()
        self.initUI()

    def initUI(self):
        self.imageEdit = QImageEdit()

        self.nextButton = QPushButton("Next")
        self.nextButton.clicked.connect(self.next_image)
        next_s = QShortcut(QKeySequence("a"), self)
        next_s.activated.connect(self.next_image)

        self.saveButton = QPushButton("Save")
        self.saveButton.clicked.connect(self.save_image)
        save_s = QShortcut(QKeySequence("s"), self)
        save_s.activated.connect(self.save_image)

        main_vbox = QVBoxLayout()
        main_vbox.addWidget(self.imageEdit)
        main_vbox.addStretch(1)
        main_vbox.addWidget(self.nextButton)
        main_vbox.addWidget(self.saveButton)
        self.setLayout(main_vbox)
        self.load_next_image()

    def save_image(self):
        rect = self.imageEdit.getImage()
        if rect is not None:
            line = self.current_image
            line += " {} {} {} {}\n".format(rect.x(), rect.y(),
                                            rect.width(), rect.height())
            with open(self.info_file, "a") as f:
                f.write(line)
        self.clear()
        self.load_next_image()

    def clear(self):
        self.imageEdit.clear()

    def next_image(self):
        line = self.current_image + " -1 -1 -1 -1\n"
        with open(self.info_file, "a") as f:
            f.write(line)
        self.clear()
        self.load_next_image()

    def load_next_image(self):
        try:
            self.current_image = next(self.next_image_gen)
        except StopIteration:
            print("end of images")
            return

        image_path = os.path.join("data", self.current_image)
        image = QImage(image_path)
        if image.size().width() > 1000 or image.size().height() > 1000:
            image = image.scaled(1000, 1000, Qt.KeepAspectRatio)
            image.save(image_path)

        self.imageEdit.setPixmap(QPixmap(image))

    def initData(self):
        folders = ["andrazpov", "robertcv"]
        all_images = []
        for u in folders:
            for i in range(20):
                for j in range(30):
                    all_images.append(u + '-{:02}'.format(i + 1) + '/{:02}.png'.format(j + 1))

        if not os.path.exists(self.info_file):
            with open(self.info_file, 'w'):
                pass
        with open(self.info_file) as f:
            observed = [l.split()[0] for l in f.readlines()]

        self.to_observe = sorted(list(set(all_images) - set(observed)))

    def get_next_image(self):
        for image in self.to_observe:
            yield image


if __name__ == '__main__':
    myQApplication = QApplication(sys.argv)
    myQMain = QMain()
    myQMain.show()
    sys.exit(myQApplication.exec_())
