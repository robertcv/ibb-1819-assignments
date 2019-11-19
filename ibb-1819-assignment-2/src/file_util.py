import random
import os.path as path


def good_images(info_file):
    info_file = path.join("data", info_file)
    with open(info_file) as f:
        return map(lambda x: x.split(";")[0],
                   filter(lambda x: x.split(";")[3].strip() == "1",
                          f.readlines()))


def good_left_images(info_file):
    info_file = path.join("data", info_file)
    with open(info_file) as f:
        return map(lambda x: x.split(";")[0],
                   filter(lambda x: x.split(";")[4].strip() == "L",
                          f.readlines()))


def good_right_images(info_file):
    info_file = path.join("data", info_file)
    with open(info_file) as f:
        return map(lambda x: x.split(";")[0],
                   filter(lambda x: x.split(";")[4].strip() == "R",
                          f.readlines()))


def good_images_files(info_file):
    return map(lambda x: path.join("data", "images", x),
               good_images(info_file))


def good_masks_files(info_file):
    return map(lambda x: path.join("data", "masks-rectangular", x),
               good_images(info_file))


def split_learn_test(info_file, learn_per=0.7, seed=0):
    random.seed(seed)

    x = list(good_images_files(info_file))
    y = list(good_masks_files(info_file))

    split_index = int(len(x) * learn_per)
    random_index = random.shuffle(list(range(len(x))))

    x_learn, x_test, y_learn, y_test = [], [], [], []

    for i in random_index[:split_index]:
        x_learn.append(x[i])
        y_learn.append(y[i])

    for i in random_index[split_index:]:
        x_test.append(x[i])
        y_test.append(y[i])

    return x_learn, y_learn, x_test, y_test


def learn_test(learn_info_file, test_info_file):
    """
    :return: x_learn, y_learn, x_test, y_test
    """

    return good_images_files(learn_info_file), \
           good_masks_files(learn_info_file), \
           good_images_files(test_info_file), \
           good_masks_files(test_info_file)


if __name__ == '__main__':
    from pprint import pprint

    pprint(list(good_images("robertcv-info.csv"))[:10])
    pprint(list(good_images_files("robertcv-info.csv"))[:10])
    pprint(list(good_masks_files("robertcv-info.csv"))[:10])
    pprint([list(lt)[:5]
            for lt in learn_test("andrazpov-info.csv", "robertcv-info.csv")])
