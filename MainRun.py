import ImageConnector
import LearningDataProvider
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Conv2D, Conv2DTranspose
from keras.layers.core import Reshape
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from pylab import imshow, show
from timeit import default_timer as timer

#
#
def automap_cnn_model(train_data):
    X_train = []
    Y_train = []
    input_data_size = int(train_data[0][0].size * 2)
    output_data_size = train_data[1][0].shape

    for image in train_data[0]:
        r = np.ndarray.flatten(np.real(image))
        i = np.ndarray.flatten(np.imag(image))
        c = np.concatenate((r, i))
        # X_train.append(c)
        X_train.append(np.reshape(c, (1, np.size(c))))

    for res in train_data[1]:
        reshaped_res = np.reshape(res, (1, 1, res.shape[0], res.shape[1]))
        Y_train.append(reshaped_res)


    # 7. Define model architecture
    model = Sequential()
    model.add(Dense(int(input_data_size / 2), input_shape=(input_data_size, ), activation='tanh'))
    model.add(Dense(int(input_data_size / 2), activation='sigmoid'))
    model.add(Reshape((1, output_data_size[0], output_data_size[1]), input_shape=(int(input_data_size / 2),)))
    model.add(Conv2D(5, (4, 4), activation='relu', padding="same", data_format="channels_first"))
    model.add(Conv2D(5, (4, 4), activation='relu', padding="same", data_format="channels_first"))
    model.add(Conv2DTranspose(1, (output_data_size[0], output_data_size[1]), padding="same", data_format="channels_first"))

    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # model.add(Conv2D(12, (4, 4), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))
    # # 8. Compile model
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model.png')
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    # # KerasVisualisationHelper.plot_all_weights(model, n=256)
    # # KerasVisualisationHelper.model_to_pic(model)
    # plot_model(model, to_file='model.png', show_shapes=True)
    score = model.evaluate(X_train[0], Y_train[0], verbose=0)
    X_train = np.ndarray(X_train)
    Y_train = np.array(Y_train)
    model.fit(X_train, Y_train,
              batch_size=32, nb_epoch=10, verbose=1)

    return model


# address = "C:\\Users\\vardi\\Documents\\Datasets\\tiny-imagenet-200\\tiny-imagenet-200\\val\\images\\"
address = "C:\\Users\\uqmbonya\\Downloads\\tiny-imagenet-200\\tiny-imagenet-200\\val\\images\\"


(img, trans) = ImageConnector.read_images(address, "JPEG", [32, 32], "k-space", True)
(train_instances), (test_instances) = LearningDataProvider.split_data(trans, img, False, .70, False)
# (x_train_instances, y_train_instances), (x_test_instances, y_test_instances) = mnist.load_data()
# (train_instances), (test_instances) = LearningDataProvider.split_data(
#     x_train_instances, y_train_instances, True, .70, True)

model = automap_cnn_model(train_instances)

i = 0
# automap_cnn_model(X_train, Y_train)



# 10. Evaluate model on test data
# score = model.evaluate(X_test, Y_test, verbose=0)
# imshow(activations_after_train)