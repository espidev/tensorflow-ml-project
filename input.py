import numpy as np
import cv2  # noqa
import os
import os.path  # noqa
import pickle
from tqdm import tqdm  # noqa
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  # noqa

IMG_SIZE = (150, 150)  # image resolution
AUGMENT_SIZE = 1  # number of images to produce for each image from base dataset


def upload_images():  # read from rawdata directory
    print("Loading Images...")
    landuses = [landuse for landuse in get_classes()]
    for i in tqdm(range(len(landuses))):
        landuses[i] = landuses[i]  # landuses.txt must end with blank line
        for name in os.listdir(f"files\\rawdata\\{landuses[i]}"):
            img = cv2.imread(
                f"files\\rawdata\\{landuses[i]}\\{name}", cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, IMG_SIZE)
            yield img, landuses[i]


def augment_images():
    if not os.path.exists('files\\augmented'):
        os.makedirs('files\\augmented')
    landuses = [landuse for landuse in get_classes()]
    for landuse in landuses:
        path = f'files\\augmented\\{landuse}'
        if not os.path.exists(path):
            os.makedirs(path)

    images = load("files\\BaseImageDataPickle")
    labels = load("files\\BaseLabelDataPickle")

    data_aug = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

    augmented_images = []
    augmented_labels = []

    for image, label in tqdm(zip(images, labels)):
        x = img_to_array(image)
        x = x.reshape((1,) + x.shape)
        count = 0
        for batch in data_aug.flow(x, batch_size=1,
                                   save_to_dir=f"files\\augmented\\{landuses[label]}",
                                   save_prefix=f"{label}",
                                   save_format='jpeg'):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            count += 1
            if count == AUGMENT_SIZE:  # number of new images to make
                break

    aug_image_pickle_file = open("files\\AugmentedImageDataPickle", "wb")
    aug_label_pickle_file = open("files\\AugmentedLabelDataPickle", "wb")
    pickle.dump(augmented_images, aug_image_pickle_file)
    pickle.dump(augmented_labels, aug_label_pickle_file)
    aug_image_pickle_file.close()
    aug_label_pickle_file.close()


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

    image_pickle_file = open("files\\BaseImageDataPickle", "wb")
    label_pickle_file = open("files\\BaseLabelDataPickle", "wb")
    pickle.dump(images, image_pickle_file)
    pickle.dump(num_labels, label_pickle_file)
    image_pickle_file.close()
    label_pickle_file.close()


def load(filename):
    file = open(filename, "rb")
    obj = pickle.load(file)
    file.close()
    return obj


def get_classes():
    file = open("files\\landuses.txt", "r")
    for landuse in file.readlines():
        yield landuse[:-1]
    file.close()


def show_images(images=[]):  # full image name OR list of images
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
# images = load("files\\ImageDataPickle")
# labels = load("files\\LabelDataPickle")
# show_images([images[0], images[9], images[4000], images[5000], images[20000]])

# Deleted augmented and rawdata folders. Extract
# new rawdata folder. Run everything commented above. Press any key to exit image windows.
# It will take a while. Let me know if anything goes wrong. Ignore below for now

# augment_images()
# aug_images = load("files\\AugmentedImageDataPickle")
# print(np.array(aug_images).shape)
# show_images = show_images(
#     [aug_images[0], aug_images[9], aug_images[4000], aug_images[5000], aug_images[20000]])
