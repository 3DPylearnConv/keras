from keras.datasets import reconstruction_dataset

import sys
import pickle
import visualization.visualize

sys.setrecursionlimit(2000)

model_filepath = "/home/jvarley/3d_conv/keras/examples/saved_models/3d_reconstruction/model_0.pkl"

dataset = reconstruction_dataset.ReconstructionDataset(patch_size=32)

f = open(model_filepath)
print model_filepath

model = pickle.load(f)

iterator = dataset.iterator(batch_size=1, num_batches=1)
x, y = iterator.next()

y_out = model._predict(x)

print "input"
visualization.visualize.visualize_3d(x[0, :, 0, :, :])
print "expected"
visualization.visualize.visualize_3d(y[0, :, 0, :, :])
print "actual"
visualization.visualize.visualize_3d(y_out)


