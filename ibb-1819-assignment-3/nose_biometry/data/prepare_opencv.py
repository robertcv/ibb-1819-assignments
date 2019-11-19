import os.path as path
import cv2 as cv


def prepare_opencv(info_file):
    with open(info_file) as f:
        images = [l.strip().split() for l in f.readlines()]
    pos_index = 1
    neg_index = 1
    pos_dat = []
    neg_dat = []
    for image in images:
        if image[0].startswith("robertcv") or image[1] == '-1':
            continue
        img = cv.imread(image[0])
        x, y, w, h = int(image[1]), int(image[2]), int(image[3]), int(image[4])

        file = "pos/{:04}.png".format(pos_index)
        cv.imwrite(file, img[y:y+h, x:x+w])
        pos_dat.append(file + " 1" + " 0" + " 0" + " {}".format(w) + " {}\n".format(h))
        pos_index += 1
        print(pos_index)

        file = "neg/{:04}.png".format(neg_index)
        cv.imwrite(file, img[:y + h, :x])
        neg_dat.append(file + "\n")
        neg_index += 1

        file = "neg/{:04}.png".format(neg_index)
        cv.imwrite(file, img[:y, x:])
        neg_dat.append(file + "\n")
        neg_index += 1

        file = "neg/{:04}.png".format(neg_index)
        cv.imwrite(file, img[y:, x + w:])
        neg_dat.append(file + "\n")
        neg_index += 1

        file = "neg/{:04}.png".format(neg_index)
        cv.imwrite(file, img[y + h:, :x + w])
        neg_dat.append(file + "\n")
        neg_index += 1

    with open("pos.dat", "w+") as f:
        f.writelines(pos_dat)

    with open("neg.dat", "w+") as f:
        f.writelines(neg_dat)


def stats(info_file):
    with open(info_file) as f:
        images = [l.strip().split() for l in f.readlines()]

    pos = len(list(filter(lambda x: x[1] != '-1', images)))
    print("{}/{} images with useful nose".format(pos, len(images)))


if __name__ == '__main__':
    stats("info.csv")
    prepare_opencv("info.csv")
