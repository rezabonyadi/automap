import numpy as np
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Dense, LeakyReLU
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.utils import plot_model
from matplotlib import pyplot as plt

import ImageConnector
import LearningDataProvider

#
#
def automap_cnn_model_build(X_train, Y_train):
    input_data_size = int(X_train[0].size)
    output_data_size = Y_train[0].shape

    # 7. Define model architecture
    model = Sequential()
    model.add(Dense(int(input_data_size / 2), input_shape=(input_data_size, ), activation='tanh'))
    model.add(Dense(int(input_data_size / 2), activation='tanh'))
    model.add(Reshape(output_data_size, input_shape=(int(input_data_size / 2.0),)))
    model.add(Conv2D(64, (5, 5), strides=1, padding="same", data_format="channels_first"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, (5, 5), strides=1, padding="same", data_format="channels_first"))
    model.add(LeakyReLU(alpha=0.3))
    # model.add(Conv2D(64, (7, 7), padding="same", data_format="channels_first"))
    model.add(Conv2DTranspose(64, (7, 7), padding="same", data_format="channels_first"))
    model.add(Conv2D(1, (7, 7), padding="same", data_format="channels_first"))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    return model


def prepare_data(data_address):
    (img, trans) = ImageConnector.read_images(data_address, "JPEG", [32, 32], "k-space", True)
    (train_instances), (test_instances) = LearningDataProvider.split_data(trans, img, False, .70, False)

    X_train = []
    Y_train = train_instances[1]
    for image in train_instances[0]:
        r = np.ndarray.flatten(np.real(image))
        i = np.ndarray.flatten(np.imag(image))
        c = np.concatenate((r, i))
        X_train.append(c)
    X_train = np.reshape(X_train, (len(X_train), len(X_train[0])))
    Y_train = np.reshape(Y_train, (len(Y_train), 1, Y_train[0].shape[0], Y_train[0].shape[1]))

    X_test = []
    Y_test = test_instances[1]
    for image in test_instances[0]:
        r = np.ndarray.flatten(np.real(image))
        i = np.ndarray.flatten(np.imag(image))
        c = np.concatenate((r, i))
        X_test.append(c)
    X_test = np.reshape(X_test, (len(X_test), len(X_test[0])))
    Y_test = np.reshape(Y_test, (len(Y_test), 1, Y_test[0].shape[0], Y_test[0].shape[1]))

    return X_train, Y_train, X_test, Y_test


def visualize_model(model, X_train, Y_train):
    # CnnVisualiser.keras_end_to_end_visualiser(model, X_train[0:1])
    # KerasVisualisationHelper.plot_layer_outputs(model, X_train[0:1], 5)
    # plot_model(model, to_file='model.png', show_shapes=True)
    # KerasVisualisationHelper.plot_all_weights(model, n=256)
    # KerasVisualisationHelper.model_to_pic(model)
    model.summary()

# address = "C:\\Users\\vardi\\Documents\\Datasets\\tiny-imagenet-200\\tiny-imagenet-200\\val\\images\\"
address = "R:\\projects\\image-net\\tiny-imagenet-200\\tiny-imagenet-200\\val\\images\\"

# (x_train_instances, y_train_instances), (x_test_instances, y_test_instances) = mnist.load_data()
# (train_instances), (test_instances) = LearningDataProvider.split_data(
#     x_train_instances, y_train_instances, True, .70, True)

X_train, Y_train, X_test, Y_test = prepare_data(address)
model = automap_cnn_model_build(X_train, Y_train)
# visualize_model(model, X_train, Y_train)
# KerasVisualisationHelper.plot_output(model, X_train[0:1], Y_train[0, 0])
model_out = model.predict(X_train[0:1])
plt.figure()
plt.title("Before training")
plt.subplot(211)
plt.imshow(Y_train[0, 0], cmap=plt.get_cmap('gray'))
plt.subplot(212)
plt.imshow(model_out[0, 0], cmap=plt.get_cmap('gray'))
plt.show()

model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
model_out = model.predict(X_train[0:1])
plt.figure()
plt.title("After training")
plt.subplot(211)
plt.imshow(Y_train[0, 0], cmap=plt.get_cmap('gray'))
plt.subplot(212)
plt.imshow(model_out[0, 0], cmap=plt.get_cmap('gray'))
plt.show()


# visualize_model(model, X_train, Y_train)

i = 0
