from matplotlib import pyplot as plt
import numpy as np
import keras.backend as K

# from PIL import Image
# import matplotlib.cbook as cbook

def get_layer_outputs(model, test_image):
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    comp_graph = [K.function([model.input] + [K.learning_phase()], [output]) for output in
                  outputs]  # evaluation functions

    # Testing
    layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]
    layer_outputs = []

    for layer_output in layer_outputs_list:
        print(layer_output[0][0].shape, end='\n-------------------\n')
        layer_outputs.append(layer_output[0][0])

    return layer_outputs


def plot_layer_outputs(model, test_image, layer_number, layer_format):
    layer_outputs = np.ndarray(get_layer_outputs(model, test_image))

    if layer_format == "channels_last":
        if len(layer_outputs.shape) == 3:
            layer_outputs = np.ndarray.transpose(layer_outputs, (2, 0, 1))
        if len(layer_outputs.shape) == 2:
            layer_outputs = np.ndarray.transpose(layer_outputs, (1, 0))

    n = layer_outputs[layer_number].shape[0]
    y_max = layer_outputs[layer_number].shape[1]
    x_max = layer_outputs[layer_number].shape[2]

    L = []
    for i in range(n):
        L.append(np.zeros((x_max, y_max)))

    for i in range(n):
        for x in range(x_max):
            for y in range(y_max):
                L[i][x][y] = layer_outputs[layer_number][i][x][y]

    return L

def keras_end_to_end_visualiser(model, test_image):
    layers = {"type": [], "images": [], "info": []}
    keras_layers = model.layers
    layer_indx = 0
    for layer in keras_layers:
        layers["type"].append(layer.__class__.__name__)
        layer_format = keras_layers[layer_indx].get_config()["data_format"]
        plot_layer_outputs(model, test_image, layer_indx, layer_format)
        layers["images"].append()
        layer_indx += 1

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
        images[num_images - 4] = dot_image # Adding dummy images between the real images when there are many filters
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
        plt.imshow(img, origin='lower', extent=place, alpha=alphas[im_ind],
                   cmap=plt.get_cmap('gray'))  # image is a small inset on axes
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
        if layer_type is "Conv2D":
            images = layers["images"][layer_indx]
            vis_layer(images, num_layers, max_images, layer_indx, overlap_x, overlap_y, sv, ev, buff_factor, layer_info)
        if layer_type is "Dense":
            image = []
            image.append(plt.imread("FCLayer.png"))
            vis_layer(image, num_layers, max_images, layer_indx, overlap_x, overlap_y, sv, ev, buff_factor, layer_info)
        layer_indx += 1

    plt.show()
