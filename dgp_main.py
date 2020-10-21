from __future__ import division
from __future__ import print_function

import time
import logging
import os
import shutil

import numpy as np
import tensorflow as tf

from Datasets import Graph
from model import SemiRFDGP

"""
seed = 2020
np.random.seed(seed)
tf.set_random_seed(seed)
"""

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', 'data', 'path of datasets')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string("label_ratio", "None", "ratio of labelled data, default split when label_ratio is None")
flags.DEFINE_integer("max_degree", 128, "")
flags.DEFINE_integer("batch_size", 256, "") # cora: 256; 
flags.DEFINE_integer("val_batch_size", 256, "")


flags.DEFINE_bool("trans_feature", False, "") # cora: False
flags.DEFINE_string("feature_dim", "[32]", "")

flags.DEFINE_string("hidden", "[32]", "hidden units of DGP") # cora: [32]
flags.DEFINE_string("n_neighbors", "[25, 10]", "")
flags.DEFINE_string("infernet", "[[64, 64], [64]]", "")
flags.DEFINE_integer("n_omega", 512, "")

flags.DEFINE_integer("sample_size", 25, "")
flags.DEFINE_float("dropout", 0.5, "")  # 0.5 
flags.DEFINE_float("weight_decay", 5e-4, "")
flags.DEFINE_float("lamb", 1.0, "scale of kl divergence")
flags.DEFINE_float("lr", 0.001, "learning rate")

flags.DEFINE_integer("pretrain_step", 500, "number of pretrain steps")
flags.DEFINE_integer("steps", 500, "steps of optimization")
flags.DEFINE_string("exp_name", "default_experiment", "experiment name")


# parameter config
label_ratio = eval(FLAGS.label_ratio)
feature_dim = eval(FLAGS.feature_dim)
hidden = eval(FLAGS.hidden)
n_neighbors = eval(FLAGS.n_neighbors)
infernet = eval(FLAGS.infernet)

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s]: %(message)s')
# create file handler which logs even debug messages
if not os.path.exists('log_dgp/{}'.format(FLAGS.dataset)):
    if not os.path.exists('log_dgp'):
        os.mkdir('log_dgp')
    os.mkdir('log_dgp/{}'.format(FLAGS.dataset))
fh = logging.FileHandler('log_dgp/{}/{}.log'.format(FLAGS.dataset, FLAGS.exp_name))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

tf.logging.set_verbosity(tf.logging.INFO)


# log parameter settings
def log_parameter_settings():
    tf.logging.info("==========Parameter Settings==========")
    tf.logging.info("dataset: {}".format(FLAGS.dataset))
    tf.logging.info("label_ratio: {}".format(label_ratio))
    tf.logging.info("======================================")


def load_data():

    # load data
    graph = Graph(FLAGS.dataset, FLAGS.data_path, label_ratio=label_ratio, max_degree=FLAGS.max_degree, batch_size=FLAGS.batch_size)
    tf.logging.info("dataset:{}, num nodes:{}, num features:{}".format(FLAGS.dataset, graph.n_nodes, graph.n_features))

    return graph


def evaluate(graph, placeholders, model, sess, test=False):

    feed_dict_val = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size, test=test, localSim=False)
    acc_val = []

    while feed_dict_val is not None:
        accuracy_val = sess.run([model.accuracy], feed_dict=feed_dict_val)
        acc_val.append(accuracy_val[0])
        feed_dict_val = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size, localSim=False)
    
    return np.mean(acc_val)

def pretrain(graph, placeholders, model, sess):

    for i in range(FLAGS.pretrain_step):
        train_feed_dict = graph.next_batch_feed_dict(placeholders, localSim=False)
        _, loss, acc_train = sess.run([model.opt_step_pt, model.loss, model.accuracy], feed_dict=train_feed_dict)
        print("pretrain step {}: loss: {:.6f}, accuracy: {:.5f}".format(i, loss, acc_train), end=", ")

        acc_val = evaluate(graph, placeholders, model, sess)
        print("Accuracy_val: {:.5f}".format(acc_val), end=", ")

        acc_test = evaluate(graph, placeholders, model, sess, test=True)
        print("Accuracy_test: {:.5f}".format(acc_test))


def train(graph, placeholders, model, sess, saver, model_path):

    max_acc_val = 0.0

    for i in range(FLAGS.steps):
        
        train_feed_dict = graph.next_batch_feed_dict(placeholders, localSim=False)
    
        sess.run(model.opt_step, feed_dict = train_feed_dict)

        if i % 5 == 0 or i == FLAGS.steps - 1:

            loss_train, re_loss, kl, acc_train_value = sess.run([model.loss, model.reconstruct_loss, 
                                        model.kl, model.accuracy], feed_dict=train_feed_dict)

            print("Epoch: {}".format(i+1), "loss: {:.5f}".format(loss_train), "llh_loss: {:.5f}".format(re_loss),
              "kl: {:.5f}".format(kl), "accuracy: {:.5f}".format(acc_train_value), end=", ")

            acc_val = evaluate(graph, placeholders, model, sess)
            print("Accuracy_val: {:.5f}".format(acc_val), end=", ")

            acc_test = evaluate(graph, placeholders, model, sess, test=True)
            print("Accuracy_test: {:.5f}".format(acc_test))
            if acc_test > max_acc_val:
                save_path = saver.save(sess, "{}/model_best.ckpt".format(model_path), global_step=i)
                print("=================successfully save the model at: {}=======================".format(save_path))
                max_acc_val = acc_test

            

if __name__ == "__main__":

    log_parameter_settings()  # log parameter settings

    # load data
    graph = load_data()

    # set placeholder
    placeholders = {
        'nodes': tf.placeholder(dtype=tf.int32, shape=[None]),
        'Y': tf.placeholder(dtype=tf.float32, shape=[None, graph.n_classes]),
        'label_mask': tf.placeholder(dtype=tf.int32, shape=[None]),
        "batch_size": tf.placeholder(tf.int32, name='batch_size')
    }
    

    hidden.append(graph.n_classes)
    assert len(hidden) == len(n_neighbors) # number of hidden layers should be equal to length of n_neighbors

    model = SemiRFDGP(placeholders, graph.feature, graph.node_neighbors, hidden, n_neighbors, infernet, FLAGS.n_omega,
                        FLAGS.trans_feature, feature_dim=feature_dim, sample_size=FLAGS.sample_size,
                        dropout=FLAGS.dropout, weight_decay=FLAGS.weight_decay, lamb=FLAGS.lamb, lr=FLAGS.lr)
    print("successfully initialized the model")

    
    saver = tf.train.Saver(max_to_keep=3)
    # initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    model_path = './log_dgp/{}/{}'.format(FLAGS.dataset, FLAGS.exp_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    
    # train the model
    pretrain(graph, placeholders, model, sess)
    train(graph, placeholders, model, sess, saver, model_path)

    # evaluate the model
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    # print(ckpt.all_model_checkpoint_paths[-1])
    # print(ckpt)

    acc_val_list = []
    acc_test_list = []

    for i in range(10):

        acc_val = evaluate(graph, placeholders, model, sess)
        acc_test = evaluate(graph, placeholders, model, sess, test=True)

        acc_val_list.append(acc_val)
        acc_test_list.append(acc_test)

    print("===============================================")
    print(acc_test_list)
    print("Accuracy_val: {:.5f}".format(np.max(acc_val_list)), end=", ")
    print("Accuracy_test: {:.5f}".format(np.max(acc_test_list)))
    print("===============================================")
    
    