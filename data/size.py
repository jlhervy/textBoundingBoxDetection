import glob
import cv2
import os
import numpy as np


dir = "imgs/"

shapes = []
i = 0
for filename in glob.glob(os.path.join(dir, "*.jpg")) :
    im = cv2.imread(filename)
    shapes.append(im.shape)
    i = i+1
    if i%20 == 0:
        print(i)

shapes = np.array(shapes)

H = shapes[:, 0]
mean_h = np.mean(H) #2355
min_h = np.min(H) #605
max_h = np.max(H) #7016

W = shapes[:, 1]
mean_w = np.mean(W) #1325
min_w = np.min(W) #436
max_w = np.max(W) #4961

mean_c = np.mean(shapes[:, 2])

HoverW = shapes[:, 0]/shapes[:, 1]
min_ratio = np.min(HoverW)

resize_H = np.round(mean_h)
padding_W = np.ceil(resize_H/min_ratio)


print(0)