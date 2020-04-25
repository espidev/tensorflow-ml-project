import numpy as np
import cv2  # noqa
import os
import os.path  # noqa
import pickle
from tqdm import tqdm  # noqa
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  # noqa


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


def augment_images():
    if not os.path.exists('files\\augmented'):
        os.makedirs('files\\augmented')
    for landuse in open("files\\landuses.txt", "r").readlines():
        path = f'files\\augmented\\{landuse[:-1]}'
        if not os.path.exists(path):
            os.makedirs(path)

    images, labels = zip(*upload_images())
    images = np.array(list(images))
    labels = np.array(list(labels))

    print(images.shape)
    print(images[0].shape)
    t = img_to_array(images[0])
    print(t.shape)
    t = t.reshape((1,) + t.shape)
    print(t.shape)
    data_aug = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

    for image, label in zip(images, labels):
        x = img_to_array(image)
        x = x.reshape((1,) + x.shape)
        count = 0
        for batch in data_aug.flow(x, batch_size=1,
                                   save_to_dir=f'files\\augmented\\{label}', save_prefix=f"{label}{count}", save_format='jpeg'):
            count += 1
            if count == 4:  # number of new images to make
                break


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


print(os.listdir("files\\rawdata"))
for name in os.listdir("files\\rawdata"):
    print(f"{name}")

# save_input()
# augment_images()
