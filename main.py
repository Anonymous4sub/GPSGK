from __future__ import division
from __future__ import print_function

import time
import logging
import os
import shutil

import numpy as np
import tensorflow as tf

from Datasets import Graph
from model import StructAwareGP

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

flags.DEFINE_integer("n_hop", 1, "") # Cora: 1
flags.DEFINE_integer("max_degree", 128, "")
flags.DEFINE_integer("path_length", 1, "") # 1
flags.DEFINE_float("path_dropout", 0.2, " ")  # 0.2

flags.DEFINE_integer("feature_dim", 64, "dimension of transformed feature") # cora: 64, citeseer:64; pubmed:64
flags.DEFINE_integer("n_samples", 780, "number of samples of omega") # cora: 780; citeseer:1000; pubmed:1000; photo:1000; computers:1000
flags.DEFINE_string("latent_layer_units", "[64, 64]", "") # cora: [64, 64]; citeseer:[64, 64]; pubmed:[64, 64]
flags.DEFINE_float("lambda1", 0.001, " ")
flags.DEFINE_float("lambda2", 1e-4, " ")  # cora: 1e-4; citeseer:1e-4; pubmed:1e-4;

flags.DEFINE_integer("batch_size", 512, "")
flags.DEFINE_integer("val_batch_size", 256, "")
flags.DEFINE_integer("steps", 1000, "steps of optimization")
flags.DEFINE_integer("pretrain_step", 100, " ") # cora: 100; citeseer:100; pubmed:100; photo:500; computers:500
flags.DEFINE_float("dropout", 0.5, "")  
flags.DEFINE_float("weight_decay", 5e-4, "")
flags.DEFINE_float("lr", 0.0005, "learning rate") # cora: 0.0005; citeseer:0.0005; pubmed:0.0005; photo:0.001; computers:0.001
flags.DEFINE_float("tau", 0.5, "") # cora: 0.5; citeseer:0.6, pubmed:0.9; photo:0.5; computers:0.5

flags.DEFINE_integer("early_stopping", 20, " ")
flags.DEFINE_string("transform", "True", "")
flags.DEFINE_string("linear_layer", "False", "")
flags.DEFINE_integer("output_dim", 16, "number of latent functions, only used when linear_layer is True")
flags.DEFINE_string("exp_name", "default_experiment", "experiment name")


# parameter config
label_ratio = eval(FLAGS.label_ratio)
if FLAGS.dataset not in ["cora", "pubmed", "citeseer"]:
    label_ratio = 0.2

if FLAGS.dataset in ["cora", "pubmed", "citeseer"]:
    if label_ratio is None:
        small = True
    elif label_ratio<0.05: # CORA的话：0.15
        small = True
    else:
        small = False
else:
    if label_ratio < 0.05: 
        small = True
    else:
        small = False

transform = eval(FLAGS.transform)
latent_layer_units = eval(FLAGS.latent_layer_units)
linear_layer = eval(FLAGS.linear_layer)

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s]: %(message)s')
# create file handler which logs even debug messages
if not os.path.exists('log/{}'.format(FLAGS.dataset)):
    if not os.path.exists('log'):
        os.mkdir('log')
    os.mkdir('log/{}'.format(FLAGS.dataset))
fh = logging.FileHandler('log/{}/{}.log'.format(FLAGS.dataset, FLAGS.exp_name))
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
    graph = Graph(FLAGS.dataset, FLAGS.data_path, label_ratio=label_ratio, n_hop=FLAGS.n_hop, max_degree=FLAGS.max_degree,
                    path_length=FLAGS.path_length, path_dropout=FLAGS.path_dropout, batch_size=FLAGS.batch_size, small=small)
    tf.logging.info("dataset:{}, num nodes:{}, num features:{}".format(FLAGS.dataset, graph.n_nodes, graph.n_features))

    return graph


def calculate_accuracy(logits, labels):

    return np.sum(np.argmax(logits, axis=1) == np.argmax(labels, axis=1)) * 100. / len(labels)

def get_psudo_label(logits, label_true, label_mask, tau=0.0):

    label = np.argmax(logits, axis=1)
    psudo_label = np.zeros_like(label_true)
    psudo_label[np.arange(label.size), label] = 1. 
    psudo_label[label_mask] = label_true[label_mask]
    
    if tau == 0.0:
        mask = np.ones(label.size, dtype=np.bool)
    else:
        preds = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        label_prob = np.max(preds, axis=1)
        mask = (label_prob >= tau)
        mask[label_mask] = True


    return psudo_label, mask


def evaluate(graph, placeholders, model, sess, test=False):

    feed_dict_val = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size, test=test)
    losses = []
    llh_losses = []
    acc_val = []

    while feed_dict_val is not None:
        loss_val, llh_loss_val, accuracy_val = sess.run([model.loss, model.reconstruct_loss, model.accuracy], feed_dict=feed_dict_val)
        losses.append(loss_val)
        llh_losses.append(llh_loss_val)
        acc_val.append(accuracy_val)
        feed_dict_val = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size)
    
    return np.mean(losses), np.mean(llh_losses), np.mean(acc_val)

def pretrain_kernel(graph, placeholders, model, sess):

    model.set_lambda(0.1, 0.01)
    for i in range(FLAGS.pretrain_step):
        train_feed_dict = graph.next_batch_feed_dict(placeholders)
        _, loss_e, acc_train = sess.run([model.opt_step_kernel, model.sim, model.accuracy], feed_dict=train_feed_dict)
        print("pretrain step {}: loss_e: {:.6f}, accuracy: {:.5f}".format(i, loss_e, acc_train))
    
    model.set_lambda(FLAGS.lambda1, FLAGS.lambda2)


def pretrain(graph, placeholders, model, sess):

    if small:
        batch_dict = graph.next_batch_feed_dict
    else:
        batch_dict = graph.next_batch_feed_dict_v2

    for i in range(FLAGS.pretrain_step):
        train_feed_dict = batch_dict(placeholders)
        _, loss_e, acc_train = sess.run([model.opt_step_e, model.loss_e, model.accuracy], feed_dict=train_feed_dict)
        print("pretrain step {}: loss_e: {:.6f}, accuracy: {:.5f}".format(i, loss_e, acc_train))


def train_iterative(graph, placeholders, model, sess, saver, model_path):

    # construct feed_dict
    # feed_dict_val = graph.val_batch_feed_dict(placeholders)
    # feed_dict_test = graph.val_batch_feed_dict(placeholders, test=True)

    # cost_val = []
    max_acc_val = 0.0

    if small:
        batch_dict = graph.next_batch_feed_dict
    else:
        batch_dict = graph.next_batch_feed_dict_v2

    for i in range(FLAGS.steps):
        
        train_feed_dict = batch_dict(placeholders)
    
        sess.run(model.opt_step_e, feed_dict = train_feed_dict)
        """
        logits = sess.run(model.logits, feed_dict=train_feed_dict)
        psudo_label, mask = get_psudo_label(logits, train_feed_dict[placeholders["Y"]], train_feed_dict[placeholders["label_mask"]], FLAGS.tau)
        train_feed_dict[placeholders["Y"]] = psudo_label
        train_feed_dict[placeholders["label_mask"]] = mask
        """
        sess.run(model.opt_step_m, feed_dict = train_feed_dict)

        """
        loss_train, re_loss, kl, sim, acc_train_value = sess.run([model.loss, model.reconstruct_loss, 
                                        model.kl, model.sim, model.accuracy], feed_dict=train_feed_dict)
        """
        sim = 0.0
        loss_train, re_loss, kl, acc_train_value = sess.run([model.loss, model.reconstruct_loss, model.kl, model.accuracy], feed_dict=train_feed_dict)

        if i % 5 == 0 or i == FLAGS.steps - 1:

            print("Epoch {}: ".format(i+1), "loss: {:.5f}, ".format(loss_train), "llh_loss: {:.5f}, ".format(re_loss),
              "kl: {:.5f}, ".format(kl), "sim: {:.5f}, ".format(sim), "Accuracy: {:.5f}".format(acc_train_value), end=", ")

            """
            feed_dict_val = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size)
            acc_val = []
            while feed_dict_val is not None:
                accuracy_val = sess.run([model.accuracy], feed_dict=feed_dict_val)
                acc_val.append(accuracy_val[0])
                feed_dict_val = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size)
            # print(acc_val)
            """
            loss_val, llh_loss_val, acc_val = evaluate(graph, placeholders, model, sess)
            print("Accuracy_val: {:.5f}".format(acc_val), end=", ")

            """
            feed_dict_test = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size, test=True)
            acc_test = []
            while feed_dict_test is not None:
                accuracy_test = sess.run([model.accuracy], feed_dict=feed_dict_test)
                acc_test.append(accuracy_test[0])
                feed_dict_test = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size, test=True)
            # print(acc_test)
            """
            loss_test, llh_loss_test, acc_test = evaluate(graph, placeholders, model, sess, test=True)
            print("loss_test: {:.5f}, llh_loss_test: {:.5f}, Accuracy_test: {:.5f}".format(loss_test, llh_loss_test, acc_test))
            if acc_val > max_acc_val:
                save_path = saver.save(sess, "{}/model_best.ckpt".format(model_path), global_step=i)
                print("=================successfully save the model at: {}=======================".format(save_path))
                max_acc_val = acc_val
        


def train(graph, placeholders, model, sess):


    # construct feed_dict
    feed_dict_val = graph.val_batch_feed_dict(placeholders)
    feed_dict_test = graph.val_batch_feed_dict(placeholders, test=True)

    for i in range(FLAGS.steps):

        t = time.time()

        train_feed_dict = graph.next_batch_feed_dict(placeholders)
        
        _, loss_value, re_loss, kl, regular, acc_train_value = sess.run([model.opt_step, model.loss, model.reconstruct_loss, model.kl,
                                                     model.sim, model.accuracy], feed_dict=train_feed_dict)
        #_, loss_value, acc_train_value = sess.run([opt_step, model.loss, model.accuracy], feed_dict=feed_dict)
        
        print("Epoch: {}".format(i+1), "Loss: {:.5f}".format(loss_value), "llh_loss: {:.5f}".format(re_loss),
              "kl: {:.5f}".format(kl), "regular: {:.5f}".format(regular), "accuracy: {:.5f}".format(acc_train_value), end=", ")

        """
        logits_val = model.predict_logits(graph.feature[graph.val_mask])
        # print(logits_val)
        logits_val_value = sess.run(logits_val, feed_dict={})
        # print(logits_val_value)
        accuracy_val = calculate_accuracy(logits_val_value, graph.y_val)
        print("Accuracy_val: {:.5f}".format(accuracy_val))
        """
        
        loss_val, accuracy_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        print("Loss_val :{:.5f}".format(loss_val), "Accuracy_val: {:.5f}".format(accuracy_val), end=", ")
        
        loss_test, accuracy_test = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_test)
        print("Loss_test :{:.5f}".format(loss_test), "Accuracy_test: {:.5f}".format(accuracy_test))

            

if __name__ == "__main__":

    log_parameter_settings()  # log parameter settings

    # load data
    graph = load_data()
    # batch_size = graph.batch_size

    # set placeholder
    placeholders = {
        'nodes': tf.placeholder(dtype=tf.int32, shape=[None]),
        'Y': tf.placeholder(dtype=tf.float32, shape=[None, graph.n_classes]),
        'label_mask': tf.placeholder(dtype=tf.int32, shape=[None]),
        'localSim': tf.placeholder(dtype=tf.float32, shape=[None, None]), 
        #'globalSim': tf.placeholder(dtype=tf.float32, shape=[None, None]),
        "batch_size": tf.placeholder(tf.int32, name='batch_size')
    }
    

    output_dim = graph.n_classes if not linear_layer else FLAGS.output_dim

    model = StructAwareGP(placeholders, graph.feature, FLAGS.feature_dim, FLAGS.n_samples, latent_layer_units, output_dim, 
                        transform_feature=transform, node_neighbors=graph.node_neighbors, linear_layer=linear_layer, 
                        lambda1=FLAGS.lambda1, lambda2 = FLAGS.lambda2, dropout=FLAGS.dropout, bias=True, 
                        act=tf.nn.relu, weight_decay=FLAGS.weight_decay, lr=FLAGS.lr)
    print("successfully initialized the model")

    saver = tf.train.Saver(max_to_keep=3)
    # initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    model_path = './log/{}/{}'.format(FLAGS.dataset, FLAGS.exp_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    
    # train the model
    # train(graph, placeholders, model, sess)
    # pretrain_kernel(graph, placeholders, model, sess)
    pretrain(graph, placeholders, model, sess)
    train_iterative(graph, placeholders, model, sess, saver, model_path)

    # evaluate the model
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    # print(ckpt.all_model_checkpoint_paths[-1])
    # print(ckpt)

    acc_val_list = []
    acc_test_list = []

    for i in range(20):

        _, _, acc_val = evaluate(graph, placeholders, model, sess)
        _, _, acc_test = evaluate(graph, placeholders, model, sess, test=True)

        acc_val_list.append(acc_val)
        acc_test_list.append(acc_test)

    print("===============================================")
    print(acc_test_list)
    print("Accuracy_val: {:.5f}".format(np.max(acc_val_list)), end=", ")
    print("Accuracy_test: {:.5f}".format(np.max(acc_test_list)))
    print("===============================================")

    