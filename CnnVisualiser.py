from matplotlib import pyplot as plt
import numpy as np
# from PIL import Image
# import matplotlib.cbook as cbook

def keras_end_to_end_visualiser(model):
    layers = {"type": [], "images": [], "info": []}
    keras_layers = model.layers
    for layer in keras_layers:

        layers["type"].append("Conv")
    i = 0

def get_place(sh, eh, sv, ev, n, i, f_x, f_y):
    x_p = (eh - sh) / (f_x * (n - 1) + 1)
    y_p = (ev - sv) / (f_y * (n - 1) + 1)
    incX = x_p * f_x
    incY = y_p * f_y

    hmin = sh + i * incX
    hmax = hmin + x_p
    vmin = 1 - (sv + i * incY)
    vmax = vmin - y_p

    return [hmin, hmax, vmin, vmax]


def vis_layer(images, num_layers, max_images, layer_indx, overlap_x, overlap_y, sv, ev, buff_factor, layer_info):
    seg_size = 1 / num_layers
    im_shapes = images[0].shape
    num_images = len(images)
    alphas = np.ones(len(images))
    if len(images) > max_images:
        dot_image = np.ones(im_shapes)
        dot_image[0, 0] = 0
        images[num_images - 4] = dot_image
        alphas[num_images - 4] = 0.5
        images[num_images - 3] = dot_image
        alphas[num_images - 3] = 0.5
        images[num_images - 2] = dot_image
        alphas[num_images - 2] = 0.5
        images[max_images - 4:num_images - 4] = []
        alphas = np.delete(alphas, range(max_images - 4, num_images - 4))

    num_imgs = min((len(images), max_images))
    sh = layer_indx * seg_size
    eh = (layer_indx + 1) * seg_size
    buff = seg_size * buff_factor
    eh -= buff
    im_ind = 0
    for img in images:
        place = get_place(sh, eh, sv, ev, num_imgs, im_ind, overlap_x, overlap_y)
        plt.imshow(img, origin='upper', extent=place, alpha=alphas[im_ind], cmap=plt.get_cmap('gray'))  # image is a small inset on axes
        plt.text(sh, 1 - (ev + 0.05), layer_info, rotation=315)
        im_ind += 1
        if im_ind >= num_imgs:
            break


def vis_layers(layers, max_images, overlap_x, overlap_y, sv, ev, buff_factor):
    num_layers = len(layers["type"])
    plt.figure(figsize=(40, 20))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for layer_indx in range(num_layers):
        layer_type = layers["type"][layer_indx]
        layer_info = layers["info"][layer_indx]
        if layer_type is "Conv":
            images = layers["images"][layer_indx]
            vis_layer(images, num_layers, max_images, layer_indx, overlap_x, overlap_y, sv, ev, buff_factor, layer_info)
        if layer_type is "Flat":
            image = []
            image.append(plt.imread("FCLayer.png"))
            vis_layer(image, num_layers, max_images, layer_indx, overlap_x, overlap_y, sv, ev, buff_factor, layer_info)
        layer_indx += 1

    plt.show()
