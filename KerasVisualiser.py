import os
import numpy as np
import numpy.ma as ma
import keras.utils.visualize_util as vu
import keras.backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#os.environ['KERAS_BACKEND'] = 'theano'
os.environ['KERAS_BACKEND'] = 'tensorflow'


class KerasVisualisationHelper(object):

    @staticmethod
    def make_mosaic(im, nrows, ncols, border=1):

        """
        From http://nbviewer.jupyter.org/github/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb

        :param im:
        :param nrows:
        :param ncols:
        :param border:
        :return:
        """


        nimgs = len(im)
        imshape = im[0].shape

        mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                                ncols * imshape[1] + (ncols - 1) * border),
                               dtype=np.float32)

        paddedh = imshape[0] + border
        paddedw = imshape[1] + border
        im
        for i in range(nimgs):

            row = int(np.floor(i / ncols))
            col = i % ncols

            mosaic[row * paddedh:row * paddedh + imshape[0],
            col * paddedw:col * paddedw + imshape[1]] = im[i]

        return mosaic

    @staticmethod
    def model_to_pic(model, file_save='model.png'):
        vu.plot(model, file_save)

    @staticmethod
    def get_weights_mosaic(model, layer_id, n=64):

        """

        :param model:
        :param layer_id:
        :param n:
        :return:
        """
        layer = model.layers[layer_id]

        # Check if this layer has weight values
        if not hasattr(layer, "W"):
            raise Exception("The layer {} of type {} does not have weights.".format(layer.name,
                                                                                    layer.__class__.__name__))

        weights = layer.W.get_value()

        # For now we only handle Conv layer like with 4 dimensions
        if weights.ndim != 4:
            raise Exception("The layer {} has {} dimensions which is not supported.".format(layer.name, weights.ndim))

        # n define the maximum number of weights to display
        if weights.shape[0] < n:
            n = weights.shape[0]

        # Create the mosaic of weights
        nrows = int(np.round(np.sqrt(n)))
        ncols = int(nrows)

        if nrows ** 2 < n:
            ncols +=1

        mosaic = KerasVisualisationHelper.make_mosaic(weights[:n, 0], nrows, ncols, border=1)

        return mosaic

    @staticmethod
    def plot_weights(model, layer_id, n=64, ax=None, **kwargs):

        """
        Plot the weights of a specific layer. ndim must be 4.
        :param model:
        :param layer_id:
        :param n:
        :param ax:
        :param kwargs:
        :return:
        """

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        layer = model.layers[layer_id]

        mosaic = KerasVisualisationHelper.get_weights_mosaic(model, layer_id, n=64)

        # Plot the mosaic
        if not ax:
            fig = plt.figure()
            ax = plt.subplot()

        im = ax.imshow(mosaic, **kwargs)
        ax.set_title("Layer #{} called '{}' of type {}".format(layer_id, layer.name, layer.__class__.__name__))

        plt.colorbar(im, ax=ax)

        return ax

    @staticmethod
    def plot_all_weights(model, n=64, **kwargs):

        """

        :param model:
        :param n:
        :param kwargs:
        :return:
        """


        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        layers_to_show = []

        for i, layer in enumerate(model.layers):
            if hasattr(layer, "W"):
                weights = layer.W.get_value()
                if weights.ndim == 4:
                    layers_to_show.append((i, layer))

        fig = plt.figure(figsize=(15, 15))

        n_mosaic = len(layers_to_show)
        nrows = int(np.round(np.sqrt(n_mosaic)))
        ncols = int(nrows)

        if nrows ** 2 < n_mosaic:
            ncols += 1

        for i, (layer_id, layer) in enumerate(layers_to_show):
            mosaic = KerasVisualisationHelper.get_weights_mosaic(model, layer_id=layer_id, n=n)

            ax = fig.add_subplot(nrows, ncols, i + 1)

            im = ax.imshow(mosaic, **kwargs)
            ax.set_title("Layer #{} called '{}' of type {}".format(layer_id, layer.name, layer.__class__.__name__))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_feature_map(model, layer_id, X, n=256, ax=None, **kwargs):
        """
        """


        layer = model.layers[layer_id]

        try:
            get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])
            activations = get_activations([X, 0])[0]
        except:
            # Ugly catch, a cleaner logic is welcome here.
            raise Exception("This layer cannot be plotted.")

        # For now we only handle feature map with 4 dimensions
        if activations.ndim != 4:
            raise Exception("Feature map of '{}' has {} dimensions which is not supported.".format(layer.name,
                                                                                                   activations.ndim))

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        fig = plt.figure(figsize=(15, 15))

        # Compute nrows and ncols for images
        n_mosaic = len(activations)
        nrows = int(np.round(np.sqrt(n_mosaic)))
        ncols = int(nrows)
        if (nrows ** 2) < n_mosaic:
            ncols += 1

        # Compute nrows and ncols for mosaics
        if activations[0].shape[0] < n:
            n = activations[0].shape[0]

        nrows_inside_mosaic = int(np.round(np.sqrt(n)))
        ncols_inside_mosaic = int(nrows_inside_mosaic)

        if nrows_inside_mosaic ** 2 < n:
            ncols_inside_mosaic += 1

        for i, feature_map in enumerate(activations):
            mosaic = KerasVisualisationHelper.make_mosaic(feature_map[:n], nrows_inside_mosaic, ncols_inside_mosaic, border=1)

            ax = fig.add_subplot(nrows, ncols, i + 1)

            im = ax.imshow(mosaic, **kwargs)
            ax.set_title("Feature map #{} \nof layer#{} \ncalled '{}' \nof type {} ".format(i, layer_id,
                                                                                            layer.name,
                                                                                            layer.__class__.__name__))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_all_feature_maps(model, X, n=256, ax=None, **kwargs):
        """
        """

        figs = []

        for i, layer in enumerate(model.layers):

            try:
                fig = KerasVisualisationHelper.plot_feature_map(model, i, X, n=n, ax=ax, **kwargs)
            except:
                pass
            else:
                figs.append(fig)

        return figs

    @staticmethod
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

    @staticmethod
    def plot_layer_outputs(model, test_image, layer_number):

        layer_outputs = KerasVisualisationHelper.get_layer_outputs(model, test_image)

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

        for img in L:
            plt.figure()
            plt.imshow(img, interpolation='nearest')

        i = 0