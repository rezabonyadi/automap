import os
from PIL import Image
import glob
import numpy as np
from skimage import color
from skimage import io



def read_images(images_addresses, extention, new_size, gray=False):
    # Read all images in the folder. You need to search all subfolders and fill the dictionary self.images
    # sub_directories = get_immediate_subdirectories(images_addresses)
    image_list = []
    for filename in glob.glob(images_addresses+'*.'+extention):  # assuming gif
        im = io.imread(filename, as_grey=gray)
        im = im.astype('float')
        if im.max() > 1:
            im = im/255.0
        image_list.append(im)
    images = []
    return images


def get_immediate_subdirectories(a_dir):
    for dirpath, dirnames, filenames in os.walk(a_dir):
        for filename in [f for f in filenames if f.endswith(".log")]:
            print(os.path.join(dirpath, filename))

    return 0
    # return [name for name in os.listdir(a_dir)
    #         if os.path.isdir(os.path.join(a_dir, name))]
