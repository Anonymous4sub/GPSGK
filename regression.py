# 2-D regression problem
import numpy as np
import tensorflow as tf
from collections import defaultdict

from layers import glorot, Layer, NeuralNet, InferenceNet
from model import GraphConvolution

scale = 28.
num_value = 300
window =1

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# randomly choose a pic
pic = x_train[np.random.randint(x_train.shape[0])] / 255.0
pic = pic.astype(np.float32)

num = pic.shape[0]**2
X = np.zeros(shape=(num, 2), dtype=np.float32)
Y = np.zeros(num, dtype=np.float32)

idx = 0
for i in range(pic.shape[0]):
    for j in range(pic.shape[0]):
        X[idx, 0] = i / scale
        X[idx, 1] = j / scale
        Y[idx] = pic[i, j]
        idx += 1

# prepare label mask
label_mask = np.zeros(num, dtype=np.bool)
label_idx = np.random.randint(0, high=num, size=num_value)
label_mask[label_idx] = True

# build node neighbors
"""
node_neighbors = np.zeros((num, 8), dtype=np.int32)
idx = 0
for i in range(pic.shape[0]):
    for j in range(pic.shape[0]):
        neighbors = list()
        for m in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if m == 0 and k == 0:
                    continue
                if i+m<0 or i+m>=28:
                    continue
                if j+k<0 or j+k>=28:
                    continue
                neighbors.append((i+m)*28+(j+k))
        if len(neighbors)<8:
            node_neighbors[idx] = np.random.choice(neighbors, size=8, replace=True)
        else:
            node_neighbors[idx] = neighbors
        idx += 1
"""
node_neighbors = np.zeros((num, 4), dtype=np.int32)
idx = 0

for i in range(pic.shape[0]):
    for j in range(pic.shape[0]):
        neighbors = list()

        if i-1>=0:
            neighbors.append((i-1)*28+j)
        if i+1<28:
            neighbors.append((i+1)*28+j)
        if j-1>=0:
            neighbors.append(i*28+(j-1))
        if j+1<28:
            neighbors.append(i*28+(j+1))

        if len(neighbors)<4:
            node_neighbors[idx] = np.random.choice(neighbors, size=4, replace=True)
        else:
            node_neighbors[idx] = neighbors
        idx += 1

# build model
feature = GraphConvolution
                


