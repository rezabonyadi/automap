import glob

import numpy as np
from skimage import io, transform


def read_images(images_addresses, extension, new_size, transformation, gray=False):
    # Read all images in the folder. You need to search all subfolders and fill the dictionary self.images
    # sub_directories = get_immediate_subdirectories(images_addresses)
    image_list = []
    transformed_list = []
    report_idx = 100
    idx = 0
    for filename in glob.glob(images_addresses+'*.'+extension):  # assuming gif
        im = io.imread(filename, as_grey=gray)
        # im = io.imread(filename)
        im = transform.resize(im, new_size, mode='reflect')
        im = im.astype('float')
        if im.max() > 1:
            im = im/255.0
        im_t = apply_transformation(im, transformation)
        image_list.append(im)
        transformed_list.append(im_t)
        if idx % report_idx == 0:
            print("Read the first %d instances" %idx)
        idx += 1
        if idx > 100:
            break
    return image_list, transformed_list


def apply_transformation(image, trans):
    transformed_image = []
    if trans == "k-space":
        transformed_image = np.fft.ifft2(image)
        # fshift = np.fft.fftshift(transformed_image)
        # magnitude_spectrum = (np.absolute(transformed_image))
        # plt.imshow(magnitude_spectrum, cmap='gray')

    return transformed_image

