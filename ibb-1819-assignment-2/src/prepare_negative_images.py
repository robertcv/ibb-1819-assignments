import urllib.request
import numpy as np
import cv2
import os

neg_path = os.path.join("data", "opencv", "neg")
neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls'


def store_raw_images(ids):
    if not os.path.exists(neg_path):
        os.makedirs(neg_path)

    pic_num = 1634
    for id in ids:
        neg_image_urls = urllib.request.\
            urlopen(neg_images_link + "?wnid=" + id).read().decode()
        print(id)
        for i in neg_image_urls.split('\n'):
            try:
                pic_file = os.path.join(neg_path, str(pic_num) + ".jpg")
                resp = urllib.request.urlopen(i.strip())
                img = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
                resized_image = cv2.resize(img, (100, 100))
                cv2.imwrite(pic_file, resized_image)
                pic_num += 1
            except Exception as e:
                print(e)


if __name__ == '__main__':
    ids = [
        "n01905661"
    ]
    store_raw_images(ids)
