import cv2
import os
import numpy as np

def prepare_dataset(filename):
    image_path = filename
    image = cv2.imread(image_path)
    image = cv2.resize(image,(128,128),interpolation=cv2.INTER_CUBIC)
    if not image is None:
        hr_image = image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

        return hr_image