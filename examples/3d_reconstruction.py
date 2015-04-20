from keras.datasets import reconstruction_dataset
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

nb_train_batches = 10
nb_test_batches = 2
nb_epoch = 2000

model = Sequential()

filter_size = 5
nb_filter_in = 1
nb_filter_out = 64
#32-5+1 = 28
model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(.5))
#out 14

filter_size = 3
nb_filter_in = nb_filter_out
nb_filter_out = 64
#14-3+1 = 12
model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(.5))
#out 6

filter_size = 3
nb_filter_in = nb_filter_out
nb_filter_out = 64
#6-3+1 = 4
model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
model.add(Dropout(.5))
#out 4

dim=4
model.add(Flatten(nb_filter_out*dim*dim*dim))
model.add(Dense(nb_filter_out*dim*dim*dim, 3000, init='normal'))
model.add(Dense(3000,4000, init='normal'))
model.add(Dense(4000, patch_size*patch_size*patch_size, init='normal'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

dataset = reconstruction_dataset.ReconstructionDataset(patch_size=patch_size)

for e in range(nb_epoch):

    train_iterator = dataset.iterator(batch_size=batch_size,
                                      num_batches=nb_test_batches,
                                      flatten_y=True)

    for b in range(nb_train_batches):
        X_batch, Y_batch = train_iterator.next()
        loss = model.train(X_batch, Y_batch)
        print 'loss: ' + str(loss)

    test_iterator = dataset.iterator(batch_size=batch_size,
                                     num_batches=nb_train_batches,
                                     flatten_y=True)

    for b in range(nb_test_batches):
        X_batch, Y_batch = test_iterator.next()
        error = model.test(X_batch, Y_batch)
        print 'error: ' + str(error)

    if e % 2 == 0:
        save_path = "saved_models/" + str(__file__).replace('.py', '') + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f = open(save_path + "model_" + str(e) + '.pkl', 'w')
        cPickle.dump(model, f)
        f.close()








