import os
from PIL import Image
import glob
import numpy as np
from skimage import color
from skimage import io, transform


def read_images(images_addresses, extention, new_size, gray=False):
    # Read all images in the folder. You need to search all subfolders and fill the dictionary self.images
    # sub_directories = get_immediate_subdirectories(images_addresses)
    image_list = []
    for filename in glob.glob(images_addresses+'*.'+extention):  # assuming gif
        im = io.imread(filename, as_grey=gray)
        im = transform.resize(im, new_size)
        im = im.astype('float')
        if im.max() > 1:
            im = im/255.0
        image_list.append(im)
    images = []
    return images


def prepare_data(images, trans):
    transformed_images = []
    for im in images:
        if trans is "k-space":
            return transformed_images

