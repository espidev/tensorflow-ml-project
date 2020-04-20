import cv2
import numpy as np

img = cv2.imread(
    "C:\\Users\\kevin\\tensorflow-ml-project\\rawdata\\airplane\\airplane00.tif", cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (256, 256))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
arr = np.array(img)

img2 = cv2.imread(
    "C:\\Users\\kevin\\tensorflow-ml-project\\rawdata\\airplane\\airplane01.tif", cv2.IMREAD_UNCHANGED)
img2 = cv2.resize(img2, (256, 256))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
arr2 = np.array(img)

stack = []
stack.append(arr)
stack.append(arr2)
stack = np.vstack(tuple(stack))
print(stack.shape)
