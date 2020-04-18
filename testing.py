import input
import numpy as np

images, labels = zip(*input.load_images())
print(images[100])
print(labels[100])

input.show_image("airplane01.tif")  # press any key to exit
