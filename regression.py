# 2-D regression problem
import numpy as np
import tensorflow as tf
from collections import defaultdict

from layers import glorot, Layer, NeuralNet, InferenceNet
from model import GraphConvolution
import matplotlib.pyplot as plt

scale = 28.
num_value = 300
window =1
feature_units = [32, 16, 16]
num_samples = [2]
kernel_units = [64, 64]
n_samples = 780
sample_size = 5
dropout = 0.5
lambda2 = 1e-6
weight_decay = 5e-4
lr = 0.001
pretrain_step = 100
step = 2000

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# randomly choose a pic
pic = x_train[123] / 255.0
pic = pic.astype(np.float32)

num = pic.shape[0]**2
X = np.zeros(shape=(num, 2), dtype=np.float32)
Y = np.zeros((num, 1), dtype=np.float32)

idx = 0
for i in range(pic.shape[0]):
    for j in range(pic.shape[0]):
        # X[idx, 0] = i / scale
        # X[idx, 1] = j / scale
        X[idx, 0] = i / scale * 2 - 1 
        X[idx, 1] = j / scale * 2 - 1 
        
        Y[idx, 0] = pic[i, j]
        idx += 1

# prepare label mask
label_mask = np.zeros(num, dtype=np.bool)
label_idx = np.random.randint(0, high=num, size=num_value)
label_mask[label_idx] = True

# build node neighbors

node_neighbors = np.zeros((num, 4), dtype=np.int32)
localSim = np.zeros((num, num), dtype=np.float32)
idx = 0

for i in range(pic.shape[0]):
    for j in range(pic.shape[0]):
        neighbors = list()

        if i-1>=0:
            neighbors.append((i-1)*28+j)
            localSim[i*28+j, (i-1)*28+j] = 1.0
        if i+1<28:
            neighbors.append((i+1)*28+j)
            localSim[i*28+j, (i+1)*28+j] = 1.0
        if j-1>=0:
            neighbors.append(i*28+(j-1))
            localSim[i*28+j, i*28+(j-1)] = 1.0
        if j+1<28:
            neighbors.append(i*28+(j+1))
            localSim[i*28+j, i*28+(j+1)] = 1.0

        if len(neighbors)<4:
            node_neighbors[idx] = np.random.choice(neighbors, size=4, replace=True)
        else:
            node_neighbors[idx] = neighbors
        idx += 1

print("==============get data====================")

# build model
with tf.variable_scope("fm"):
    
    feature = GraphConvolution(X, node_neighbors, feature_units, num_samples, num, dropout=dropout)(np.arange(num))
    # feature = tf.identity(X)
    feature_dim = feature.get_shape().as_list()[1]
    # feature = NeuralNet(2, [feature_units[0]], dropout=dropout, act=tf.nn.relu)(X)
    # feature = GraphConvolution(feature, node_neighbors, [feature_units[1]], num_samples, num, dropout=dropout)(np.arange(num))
    feature = NeuralNet(feature_dim, feature_units[1:], dropout)(feature)
    
    
feature_dim = feature.get_shape().as_list()[1]

with tf.variable_scope("ikn"):
    kernelnet = InferenceNet(1, feature_dim, kernel_units, dropout=dropout)
    context_weight = glorot([n_samples, n_samples])

epsilon = np.random.normal(0.0, 1.0, [n_samples, 1]).astype(np.float32)
eps = np.random.normal(0.0, 1.0, [sample_size, n_samples, feature_dim]).astype(np.float32)
b = np.random.uniform(0.0, 2*np.pi, [1, n_samples]).astype(np.float32)

with tf.variable_scope("pos"):
    W_mu = glorot([1, n_samples])
    W_logstd = tf.Variable(-0.01, dtype=tf.float32) * tf.ones([1, n_samples], dtype=tf.float32)

Omega_mu, Omega_logstd = kernelnet(epsilon)
Omega = Omega_mu + eps * tf.math.exp(Omega_logstd)  # sample_size, n_samples, feature_dim
Omega = tf.reduce_mean(Omega, axis=0)

transform = tf.matmul(feature, Omega, transpose_b=True) # N, n_samples
transform = np.sqrt(2. / n_samples) * tf.math.cos(2*np.pi*transform + b)
kernelfeatures = tf.cast(transform, tf.float32)

# obtain parameters of the linear mapping
u = np.random.normal(0.0, 1.0, [sample_size, 1, n_samples])
W = W_mu + u * tf.math.exp(W_logstd)
W = tf.reduce_mean(W, axis=0)

kernelfeatures = tf.nn.dropout(kernelfeatures, keep_prob=1-dropout)
output = tf.matmul(kernelfeatures, W, transpose_b=True)


# ============================= construct loss =====================================

reconstruct_loss = tf.reduce_mean(tf.square(output-Y)*tf.cast(label_mask, tf.float32))

scale = 1. / tf.reduce_sum(tf.cast(label_mask, tf.float32))
kl = scale * 0.5 * tf.reduce_mean(tf.reduce_sum(tf.math.square(W_mu) + tf.math.square(tf.math.exp(W_logstd)) - 2*W_logstd - 1, axis=1))

context = tf.matmul(kernelfeatures, context_weight)
sim_contex = tf.matmul(kernelfeatures, context, transpose_b=True)
mask_not_neighbor = tf.equal(localSim, 0.0)
pos_neighbor = tf.nn.dropout(tf.boolean_mask(sim_contex, tf.math.logical_not(mask_not_neighbor)), keep_prob=0.5)
neg_neighbor = tf.nn.dropout(tf.boolean_mask(sim_contex, mask_not_neighbor), keep_prob=0.5)
sim_neighbor = -1 * tf.reduce_mean(tf.math.log(1e-6 + tf.nn.sigmoid(pos_neighbor))) - tf.reduce_mean(tf.math.log(1e-6 + tf.nn.sigmoid(-1 * neg_neighbor)))

sim = lambda2 * sim_neighbor

fm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="fm")
kn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ikn")
ps_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pos")
tf_vars = fm_vars + kn_vars

l2_loss = 0
for var in tf_vars:
    l2_loss += tf.nn.l2_loss(var)
l2_loss = weight_decay * l2_loss

loss_e = reconstruct_loss + kl + l2_loss
loss_m = reconstruct_loss + sim + l2_loss


optimizer_e = tf.train.AdamOptimizer(lr)
grads_and_vars_e = optimizer_e.compute_gradients(loss_e, var_list= fm_vars+ ps_vars)
clipped_grads_and_vars_e = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                            for grad, var in grads_and_vars_e]
opt_step_e = optimizer_e.apply_gradients(clipped_grads_and_vars_e)

optimizer_m = tf.train.AdamOptimizer(lr)
grads_and_vars_m = optimizer_m.compute_gradients(loss_m, var_list= fm_vars + kn_vars)
clipped_grads_and_vars_m = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                            for grad, var in grads_and_vars_m]
opt_step_m = optimizer_m.apply_gradients(clipped_grads_and_vars_m)

print("=================build model=========================")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# pretrain the model

for i in range(pretrain_step):
    _, loss_e_value, loss_m_value, re_loss = sess.run([opt_step_e, loss_e, loss_m, reconstruct_loss])
    print("Pretrain: loss_e: {:.5f}, loss_m: {:.5f}, MSE: {:.5f}".format(loss_e_value, loss_m_value, re_loss)) 

for i in range(step):

    sess.run(opt_step_e)
    sess.run(opt_step_m)

    loss_e_value, loss_m_value, re_loss = sess.run([loss_e, loss_m, reconstruct_loss])
    print("Step {}: loss_e: {:.5f}, loss_m: {:.5f}, MSE: {:.5f}".format(i+1, loss_e_value, loss_m_value, re_loss)) 

    
# visualize the pic
pic_re = sess.run(output)

Y = Y.reshape(num)
pic_mask = Y * 255
pic_mask[~label_mask] = 0
pic_mask[np.logical_and(label_mask, Y==0)] = 100
pic_mask = pic_mask.reshape((28, 28))
pic_mask = pic_mask.astype(np.uint8)

pic = pic * 255
pic = pic.astype(np.uint8) 

pic_re = pic_re.flatten().reshape((28, 28))* 255.
pic_re = pic_re.astype(np.uint8)

# 
fig = plt.figure()

"""
data = [pic_mask, pic, pic_re]
for (i, p) in enumerate(data):
    a = fig.add_subplot(1, len(data), i+1)
    plt.imshow(p, cmap='gray')
    plt.axis('off')

"""
a1 = fig.add_subplot(1, 3, 1)
plt.imshow(pic_mask, cmap='Blues')
plt.axis('off')

a1 = fig.add_subplot(1, 3, 2)
plt.imshow(pic, cmap='gray')
plt.axis('off')

a2 = fig.add_subplot(1, 3, 3)
plt.imshow(pic_re, cmap='gray')
plt.axis('off')


plt.show()