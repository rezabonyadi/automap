import ImageConnector
import LearningDataProvider
from keras.datasets import mnist

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import MaxPooling2D, Conv2D
# from keras.utils import np_utils, plot_model
#
#
# def automap_cnn_model(X_train, Y_train):
#     # 7. Define model architecture
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28),
#                      data_format="channels_first"))
#     model.add(Conv2D(12, (4, 4), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation='softmax'))
#     # 8. Compile model
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     model.summary()
#     # plot_model(model, to_file='model.png')
#     # SVG(model_to_dot(model).create(prog='dot', format='svg'))
#     # KerasVisualisationHelper.plot_all_weights(model, n=256)
#     # KerasVisualisationHelper.model_to_pic(model)
#     plot_model(model, to_file='model.png', show_shapes=True)
#     model.fit(X_train, Y_train,
#               batch_size=32, nb_epoch=10, verbose=1)


address = "C:\\Users\\vardi\\Documents\\Datasets\\tiny-imagenet-200\\tiny-imagenet-200\\val\\images\\"
# address = "C:\\Users\\uqmbonya\\Downloads\\tiny-imagenet-200\\tiny-imagenet-200\\val\\images\\"


# (img, trans) = ImageConnector.read_images(address, "JPEG", [64, 64], "k-space", True)
# (train_instances), (test_instances) = LearningDataProvider.split_data(img, trans, True, .70, True)
(x_train_instances, y_train_instances), (x_test_instances, y_test_instances) = mnist.load_data()
(train_instances), (test_instances) = LearningDataProvider.split_data(x_train_instances, y_train_instances, True, .70,
                                                                      True)

i = 0
# automap_cnn_model(X_train, Y_train)



# 10. Evaluate model on test data
# score = model.evaluate(X_test, Y_test, verbose=0)
# imshow(activations_after_train)