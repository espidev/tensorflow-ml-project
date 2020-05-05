import numpy as np
import cv2  # noqa
import os
import os.path  # noqa
import msgpack  # noqa
from tqdm import tqdm  # noqa
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  # noqa


# retrieve landuse catgories
def get_classes():
    file = open("files/landuses.txt", "r")
    for landuse in file.readlines():
        yield landuse[:-1]  # landuses.txt must end with blank line
    file.close()


# read from rawdata directory
def upload(img_size=(150, 150), dir="rawdata"):
    print("Loading Images...")
    landuses = [landuse for landuse in get_classes()]
    for i in tqdm(range(len(landuses))):
        for name in os.listdir(f"files/{dir}/{landuses[i]}"):
            img = cv2.imread(
                f"files/{dir}/{landuses[i]}/{name}", cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, img_size)
            yield img, landuses[i]


# pickling images and labels to prevent re-uploading from rawdata every time
def serialize(name="Base", dir="rawdata", img_size=(150, 150)):
    images, labels = zip(*upload(img_size=img_size, dir=dir))
    images = np.array(list(images))
    labels = np.array(list(labels))

    num_labels = []
    current = labels[0]
    index = 0
    for label in labels:
        if label != current:
            index += 1
            current = label
        num_labels.append(index)
    num_labels = np.array(num_labels)

    with open(f"files/{name}CompressedData.npz", "wb") as file:
        np.savez_compressed(file, images=images, labels=num_labels)


# retrieve serialized images
def load(filename):
    with open(f"files/{filename}.npz", "rb") as file:
        arr = np.load(file)
        return arr["images"], arr["labels"]


# use keras image augmentation to decrease overfitting
def augment(new=1):
    if not os.path.exists('files/augmented'):
        os.makedirs('files/augmented')
    landuses = [landuse for landuse in get_classes()]
    for landuse in landuses:
        path = f'files/augmented/{landuse}'
        if not os.path.exists(path):
            os.makedirs(path)

    images, labels = load('files/BaseCompressedData.npz')

    data_aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

    for image, label in tqdm(zip(images, labels)):
        x = img_to_array(image)
        x = x.reshape((1,) + x.shape)
        count = 0
        for batch in data_aug.flow(x, batch_size=1,
                                   save_to_dir=f"files/augmented/{landuses[label]}",
                                   save_prefix=f"{label}",
                                   save_format='jpeg'):
            count += 1
            if count == new:
                break

    serialize(name="Augmented", dir="augmented")


# display list of images
def show_images(images=[]):
    if (len(images) == 0):
        print("Error: Empty Image")
    else:
        count = 1
        for image in images:
            cv2.imshow(f"ImageWindow{count}", image)
            count += 1
        cv2.waitKey(0)  # press any key to exit
        cv2.destroyAllWindows()


# turn images into grayscale
def grayscale():
    if not os.path.exists('files/grayscales'):
        os.makedirs('files/grayscales')
    landuses = [landuse for landuse in get_classes()]
    for landuse in landuses:
        path = f'files/grayscales/{landuse}'
        if not os.path.exists(path):
            os.makedirs(path)

    images, labels = load('files/BaseCompressedData.npz')
    count = 0
    for image, label in tqdm(zip(images, labels)):
        cv2.imwrite(
            f"files/grayscales/{landuses[label]}/{landuses[label]}{count}.jpeg", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        count += 1
    serialize(name="Grays", dir="grayscales")


# save_input()
# images = load("files/BaseImageDataPickle")
# labels = load("files/BaseLabelDataPickle")
# show_images([images[0], images[9], images[4000], images[5000], images[20000]])

# Deleted augmented and rawdata folders. Extract
# new rawdata folder. Run everything commented above. Press any key to exit image windows.
# It will take a while. Let me know if anything goes wrong.

# augment_images()
# aug_images = load("files/AugmentedImageDataPickle")
# print(np.array(aug_images).shape)
# show_images = show_images(
#     [aug_images[i] for i in range(0, 24000, 1000)])

# augment images works now. Uncomment above and run.


# load and resize VGG19 images
# save_input(name="VGG19")

# vggimgs = load("files/VGG19ImageDataPickle")
# vggimgs = np.array(vggimgs)
# print(vggimgs.shape)
