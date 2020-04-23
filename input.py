import numpy as np
import cv2  # noqa
import os
import os.path  # noqa
import pickle
from tqdm import tqdm  # noqa


def upload_images():  # read from rawdata directory
    print("Loading Images...")
    file = open("files\\landuses.txt", "r")
    landuses = file.readlines()
    for i in tqdm(range(len(landuses))):
        landuses[i] = landuses[i][:-1]  # landuses.txt must end with blank line
        # print(landuses[i])
        for name in os.listdir(f"files\\rawdata\\{landuses[i]}"):
            img = cv2.imread(
                f"files\\rawdata\\{landuses[i]}\\{name}", cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (256, 256))
            yield img, landuses[i]


def save_input():  # pickling images and labels to prevent reloading from rawdata every time
    images, labels = zip(*upload_images())
    images = list(images)
    labels = list(labels)

    num_labels = []
    current = labels[0]
    index = 0
    for label in labels:
        if label != current:
            index += 1
            current = label
        num_labels.append(index)

    image_pickle_file = open("files\\UCMercedImages", "wb")
    label_pickle_file = open("files\\UCMercedLabels", "wb")

    pickle.dump(images, image_pickle_file)
    pickle.dump(num_labels, label_pickle_file)

    image_pickle_file.close()
    label_pickle_file.close()


def load(filename):
    file = open(filename, "rb")
    obj = pickle.load(file)
    file.close()
    return obj


def show_images(str="", images=[]):  # full image name OR list of images
    if (len(str) > 0):
        name = str[:-6]
        img = cv2.imread(
            f"files\\rawdata\\{name}\\{str}", cv2.IMREAD_UNCHANGED)
        cv2.imshow(str, img)
        cv2.waitKey(0)
    else:
        if (len(images) == 0):
            print("Error: Empty Image")
        else:
            count = 1
            for image in images:
                cv2.imshow(f"ImageWindow{count}", image)
                count += 1
            cv2.waitKey(0)  # press any key to exit
            cv2.destroyAllWindows()


def grayscale(images):
    grays = []
    for image in images:
        grays.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return np.array(grays)

# save_input()
