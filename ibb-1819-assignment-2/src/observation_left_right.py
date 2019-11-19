import os
import sys
from PyQt5.QtWidgets import QLabel, QWidget, \
    QPushButton, QHBoxLayout, QVBoxLayout, QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class QMain(QWidget):
    def __init__(self, *args, bitbucket_name='name', **kwargs):
        super(QMain, self).__init__(*args, **kwargs)
        self.bitbucket_name = bitbucket_name
        self.info_file = os.path.join('data', self.bitbucket_name + '-info.csv')
        self.subjects = []
        self.annotated_subjects = []
        self.current_subject = []
        self.initFolders()
        self.next_subject = self.get_next_subjact()
        self.initUI()
        self.load_image()

    def initUI(self):
        self.image = QLabel()
        self.ear_image = QLabel()

        image_hbox = QHBoxLayout()
        image_hbox.addWidget(self.image)

        self.notButton = QPushButton("Right")
        self.notButton.clicked.connect(self.save_choise_not)

        self.okButton = QPushButton("Left")
        self.okButton.clicked.connect(self.save_choise_ok)

        hbox = QHBoxLayout()
        hbox.addWidget(self.notButton)
        hbox.addWidget(self.okButton)

        main_vbox = QVBoxLayout()
        main_vbox.addStretch(1)
        main_vbox.addLayout(image_hbox)
        main_vbox.addStretch(1)
        main_vbox.addLayout(hbox)
        self.setLayout(main_vbox)

    def load_image(self):
        self.image.clear()
        try:
            self.current_subject = next(self.next_subject)
        except StopIteration:
            self.save()
            return

        image_path = os.path.join("data", "images",
                                      self.current_subject[0])
        image = QImage(image_path)
        if image.size().width() > 500 or image.size().height() > 500:
            image = image.scaled(500, 500, Qt.KeepAspectRatio)

        self.image.setPixmap(QPixmap(image))

    def save_choise_not(self):
        self.current_subject.append("R")
        self.annotated_subjects.append(self.current_subject)
        self.load_image()

    def save_choise_ok(self):
        self.current_subject.append("L")
        self.annotated_subjects.append(self.current_subject)
        self.load_image()

    def save(self):
        with open(self.info_file+".tmp", 'w') as f:
            f.writelines(map(lambda x: ";".join(map(str, x)) + "\n",
                             self.annotated_subjects))

    def initFolders(self):
        if os.path.isfile(self.info_file):
            with open(self.info_file) as info_f:
                idata = info_f.readlines()
            self.subjects = list(map(lambda x: x[:-1].split(';'), idata))

    def get_next_subjact(self):
        for s in self.subjects:
            if len(s) == 4:
                if s[-1] == "1":
                    yield s
                else:
                    self.annotated_subjects.append(s + ["-"])

            else:
                self.annotated_subjects.append(s)


if __name__ == '__main__':
    BITBUCKET_NAME = "andrazpov"
    myQApplication = QApplication(sys.argv)
    myQMain = QMain(bitbucket_name=BITBUCKET_NAME)
    myQMain.show()
    sys.exit(myQApplication.exec_())
