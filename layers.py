# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:16:26 2020

@author: fangjy
"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, bias=True, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        with tf.variable_scope(self.name + '_vars'):
            self.vars["weights"] = glorot([self.input_dim, self.output_dim], name="weights")
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        x = inputs
        x = tf.nn.dropout(x, keep_prob = 1-self.dropout)
        # transform
        output = tf.matmul(x, self.vars['weights'])
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
    
    def get_vars(self):
        variables = [self.vars["weights"]]
        if self.bias:
            variables.append(self.vars["bias"])
        
        return variables


class NeuralNet(Layer):

    """feed-forward neural networks"""

    def __init__(self, input_dim, latent_layers_dim, dropout=0.5, act=tf.nn.relu, name=None, **kwargs):

        super(NeuralNet, self).__init__(**kwargs)

        self.act = act 
        self.all_layers_dim = [input_dim]
        self.all_layers_dim.extend(latent_layers_dim)

        self.hidden_layers = []
        for in_dim, out_dim in zip(self.all_layers_dim[:-1], self.all_layers_dim[1:]):
            self.hidden_layers.append(Dense(in_dim, out_dim, dropout=dropout, act=self.act, bias=True))

    def _call(self, inputs):

        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        
        return x

class InferenceNet(Layer):

    def __init__(self, input_dim, output_dim, latent_layers_dim, dropout=0.0, act=tf.nn.relu, name=None, **kwargs):

        super(InferenceNet, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.act = act

        self.hidden_layers = NeuralNet(input_dim, latent_layers_dim, dropout, act=self.act)

        self.mu_layer = Dense(latent_layers_dim[-1], self.output_dim, dropout=dropout, act=lambda x: x , bias=False)
        self.logstd_layer = Dense(latent_layers_dim[-1], self.output_dim, dropout=dropout, act=lambda x: x, bias=False)

    def _call(self, inputs):

        x = inputs
        
        """
        for layer in self.hidden_layers:
            x = layer(x)
        """
        x = self.hidden_layers(x)

        mu = self.mu_layer(x)
        logstd = self.logstd_layer(x)

        return mu, logstd


class Constant(Layer):

    def __init__(self, input_dim,  **kwargs):
        super(Constant, self).__init__(**kwargs)

        #self.W = tf.eye(input_dim, dtype=tf.float32)
    
    def _call(self, inputs):
        #return tf.matmul(inputs, self.W)
        return inputs
    
    def get_vars(self):
        variables = []
        return variables


class GraphConvolution(Layer):

    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim], name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        # x = tf.sparse_tensor_dense_matmul(self.adj, x)
        x = tf.matmul(self.adj, x)
        outputs = self.act(x)
        return outputs
    
    def get_vars(self):
        return [self.vars["weights"]]


class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
            
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim], name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):

        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        means = tf.reduce_mean(tf.concat([neigh_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)
       
        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class RFFAggregator(Layer):

    def __init__(self, input_dim, output_dim, latent_units, n_omega, dropout=0.5, act=tf.nn.relu, sample_size=1, name=None, concat=False, **kwargs):
            
        super(RFFAggregator, self).__init__(**kwargs)

        self.n_omega = n_omega
        self.dropout = dropout
        self.act = act
        self.sample_size = sample_size
        self.concat = concat
        
        with tf.variable_scope("{}_inferencenet".format(name)):
            self.inference = InferenceNet(input_dim, input_dim, latent_units, dropout=self.dropout, act=self.act)
        
        # self.mu_omega = glorot([1, input_dim])
        # self.logstd_omega = glorot([1, input_dim])

        self.eps = np.random.normal(0.0, 1.0, [self.sample_size, n_omega, input_dim]).astype(np.float32)
        self.b = np.random.uniform(0.0, 2*np.pi, [1, n_omega]).astype(np.float32)

        self.W_mu = glorot([n_omega, output_dim])
        self.W_logstd = glorot([n_omega, output_dim])
        self.u = np.random.normal(0.0, 1.0, [self.sample_size, n_omega, output_dim]).astype(np.float32)
    

    def _call(self, inputs):

        self_vecs, neigh_vecs = inputs
        neigh_vecs = tf.nn.dropout(neigh_vecs, keep_prob = 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, keep_prob = 1-self.dropout)

        self_vecs = tf.expand_dims(self_vecs, axis=1)
        all_vecs = tf.concat([neigh_vecs, self_vecs], axis=1) # [n_self_vecs, n_neighbors+1, input_dim]
        all_means = tf.expand_dims(tf.reduce_mean(all_vecs, axis=[0, 1]), axis=0)  # [1, input_dim]

        self.mu_omega, self.logstd_omega = self.inference(all_means)  # [1, input_dim]
        omega = self.mu_omega + self.eps * tf.math.exp(self.logstd_omega)  # [sample_size, n_omega, input_dim]
        omega = tf.reduce_mean(omega, axis=0)  # [n_omega, input_dim]

        transform = tf.matmul(all_vecs, omega, transpose_b=True)
        transform = np.sqrt(2. / self.n_omega) * tf.math.cos(2*np.pi*transform + self.b)
        kernelfeatures = tf.cast(transform, tf.float32)

        W = tf.reduce_mean(self.W_mu + self.u * tf.math.exp(self.W_logstd), axis=0)
        output = tf.reduce_mean(tf.matmul(kernelfeatures, W), axis=1)  

        return output

    
    def obtain_KL_prior(self):
        
        kl_omega = 0.5 * tf.reduce_sum(tf.math.square(self.mu_omega) + tf.math.square(tf.math.exp(self.logstd_omega)) \
                                        - 2 * self.logstd_omega - 1)
        kl_W = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.math.square(self.W_mu) + tf.math.square(tf.math.exp(self.W_logstd)) \
                                    - 2*self.W_logstd - 1, axis=1))
        return kl_omega + kl_W