from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# # the data, shuffled and split between tran and test sets
# (X_train, y_train), (X_test, y_test) = cifar10.load_data(test_split=0.1)
# print X_train.shape[0], 'train samples'
# print X_test.shape[0], 'test samples'
#
# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution3D(nb_filter=4, stack_size=1, nb_row=3, nb_col=3, nb_depth=3, border_mode='valid'))
model.add(Flatten(4))
model.add(Dense(4, nb_classes, init='normal'))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# if not data_augmentation:
#     print "Not using data augmentation or normalization"
#
#     X_train = X_train.astype("float32")
#     X_test = X_test.astype("float32")
#     X_train /= 255
#     X_test /= 255
#     model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=10)
#     score = model.evaluate(X_test, Y_test, batch_size=batch_size)
#     print 'Test score:', score

# else:
#     print "Using real time data augmentation"
#
#     # this will do preprocessing and realtime data augmentation
#     datagen = ImageDataGenerator(
#         featurewise_center=True, # set input mean to 0 over the dataset
#         samplewise_center=False, # set each sample mean to 0
#         featurewise_std_normalization=True, # divide inputs by std of the dataset
#         samplewise_std_normalization=False, # divide each input by its std
#         zca_whitening=False, # apply ZCA whitening
#         rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True, # randomly flip images
#         vertical_flip=False) # randomly flip images
#
#     # compute quantities required for featurewise normalization
#     # (std, mean, and principal components if ZCA whitening is applied)
#     datagen.fit(X_train)

import numpy as np

x = np.zeros((32, 3, 1, 3, 3))

# self.W_shape = (nb_filter, nb_depth, stack_size, nb_row, nb_col)
# input_shape = (n_input_samples, input_z_dim, n_input_channels, input_x_dim, input_y_dim)
# filter_shape = (n_filter_out_channels,
#                 filter_z_dim,
#                 n_filter_in_channels,
#                 filter_x_dim,
#                 filter_y_dim)

y = np.zeros((32, 10))


import IPython
IPython.embed()
# for e in range(nb_epoch):
#     print '-'*40
#     print 'Epoch', e
#     print '-'*40
#     print "Training..."
#     # batch train with realtime data augmentation
#     progbar = generic_utils.Progbar(X_train.shape[0])
#     for X_batch, Y_batch in datagen.flow(X_train, Y_train):
#         loss = model.train(X_batch, Y_batch)
#         progbar.add(X_batch.shape[0], values=[("train loss", loss)])
#
#     print "Testing..."
#     # test time!
#     progbar = generic_utils.Progbar(X_test.shape[0])
#     for X_batch, Y_batch in datagen.flow(X_test, Y_test):
#         score = model.test(X_batch, Y_batch)
#         progbar.add(X_batch.shape[0], values=[("test loss", score)])








