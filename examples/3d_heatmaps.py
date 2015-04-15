from keras.datasets import point_cloud_hdf5_dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

import numpy as np
import os
import cPickle
import sys

sys.setrecursionlimit(2000)

batch_size = 16
patch_size = 32

nb_train_batches = 8
nb_test_batches = 2
nb_classes = 32
nb_epoch = 2000

model = Sequential()
filter_size = 5
nb_filter_in = 1
nb_filter_out = 32
#32-5+1 = 28
model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(.5))

filter_size = 3
nb_filter_in = nb_filter_out
nb_filter_out = 64
#14-3+1 = 12
model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(.5))

filter_size = 6
nb_filter_in = nb_filter_out
nb_filter_out = nb_classes
#6-6+1
model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
model.add(Flatten(nb_filter_out))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

hdf5_filepath = '/srv/3d_conv_data/training_data/contact_and_potential_grasps-3_23_15_34-3_23_16_35.h5'
topo_view_key = 'rgbd'
y_key = 'grasp_type'

dataset = point_cloud_hdf5_dataset.PointCloud_HDF5_Dataset(topo_view_key,
                                                   y_key,
                                                   hdf5_filepath,
                                                   patch_size)


for e in range(nb_epoch):

    train_iterator = dataset.iterator(batch_size=batch_size,
                            num_batches=nb_train_batches,
                            mode='even_shuffled_sequential')

    for b in range(nb_train_batches):
        X_batch, Y_batch = train_iterator.next()
        Y_batch = np.where(np.abs(Y_batch) > 0, 1.0, 0.0)
        loss = model.train(X_batch, Y_batch)
        print 'loss: ' + str(loss)

    test_iterator = dataset.iterator(batch_size=batch_size,
                            num_batches=nb_test_batches,
                            mode='even_shuffled_sequential')

    for b in range(nb_test_batches):
        X_batch, Y_batch = test_iterator.next()
        Y_batch = np.where(np.abs(Y_batch) > 0, 1.0, 0.0)
        error = model.test(X_batch, Y_batch)
        print 'error: ' + str(error)

    if e % 2 == 0:
        save_path = "saved_models/" + str(__file__).replace('.py', '') + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f = open(save_path + "model_" + str(e) + '.pkl', 'w')
        cPickle.dump(model, f)
        f.close()









