import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2  # noqa
import os
import os.path
from tqdm import tqdm  # noqa


def load_images():
    print("Loading Images...")
    file = open("landuses.txt", "r")
    landuses = file.readlines()
    for i in tqdm(range(len(landuses))):
        landuses[i] = landuses[i][:-1]  # landuses.txt must end with blank line
        # print(landuses[i])
        for name in os.listdir(f"rawdata\\{landuses[i]}"):
            img = cv2.imread(
                f"rawdata\\{landuses[i]}\\{name}", cv2.IMREAD_UNCHANGED)
            yield img, landuses[i]


def show_image(str):  # full image name
    name = str[:-6]
    img = cv2.imread(
        f"rawdata\\{name}\\{str}", cv2.IMREAD_UNCHANGED)
    cv2.imshow(str, img)
    cv2.waitKey(0)
