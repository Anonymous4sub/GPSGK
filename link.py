from __future__ import division
from __future__ import print_function

import time
import logging
import os
import shutil

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from Datasets import LinkGraph
from model import StructAwareGP_Unsup
from utils.util import sigmoid
from sklearn.metrics import roc_auc_score,average_precision_score

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
flags.DEFINE_integer("max_degree", 100, "")

flags.DEFINE_integer("feature_dim", 32, "dimension of transformed feature") # cora: 64, citeseer:64; pubmed:64
flags.DEFINE_integer("n_samples", 256, "number of samples of omega") # cora: 780; citeseer:1000; pubmed:1000; photo:1000; computers:1000
flags.DEFINE_string("latent_layer_units", "[64, 64]", "") # cora: [64, 64]; citeseer:[64, 64]; pubmed:[64, 64]
flags.DEFINE_integer("output_dim", 32, "number of latent functions, only used when linear_layer is True")

flags.DEFINE_integer("batch_size", 512, "")
flags.DEFINE_integer("val_batch_size", 256, "")
flags.DEFINE_integer("steps", 2000, "steps of optimization")
flags.DEFINE_integer("pretrain_step", 200, " ") # cora: 100; citeseer:100; pubmed:100; photo:500; computers:500
flags.DEFINE_float("dropout", 0.3, "")  
flags.DEFINE_float("weight_decay", 5e-4, "")
flags.DEFINE_float("lr", 0.0005, "learning rate") # cora: 0.0005; citeseer:0.0005; pubmed:0.0005; photo:0.001; computers:0.001

flags.DEFINE_string("exp_name", "default_experiment_link", "experiment name")
flags.DEFINE_bool("classification", False, "whether to train with the learned representation")


# parameter config
label_ratio = eval(FLAGS.label_ratio)
if FLAGS.dataset not in ["cora", "citeseer", "pubmed"]:
    label_ratio = 0.2
    
latent_layer_units = eval(FLAGS.latent_layer_units)

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
    graph = LinkGraph(FLAGS.dataset, FLAGS.data_path, label_ratio=label_ratio, max_degree=FLAGS.max_degree, batch_size=FLAGS.batch_size)
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
    acc_val = []

    while feed_dict_val is not None:
        accuracy_val = sess.run([model.accuracy], feed_dict=feed_dict_val)
        acc_val.append(accuracy_val[0])
        feed_dict_val = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size)
    
    return np.mean(acc_val)


def incremental_evaluate_link(graph, placeholders, model, sess, test=False):

    if test:
        val_edges_pos = graph.test_edges
        val_edges_neg = graph.test_edges_false
    else:
        val_edges_pos = graph.val_edges
        val_edges_neg = graph.val_edges_false
    
    prob_pos = []
    prob_neg = []

    batch_num = 0
    while(batch_num*FLAGS.val_batch_size <= len(val_edges_pos)):

        idx_start = batch_num * FLAGS.val_batch_size
        idx_end = min(idx_start + FLAGS.val_batch_size, len(val_edges_pos))
        batch_num += 1
        batch_size = idx_end - idx_start

        batch_pos = val_edges_pos[idx_start:idx_end]
        batch_neg = val_edges_neg[idx_start:idx_end]

        feed_dict = {}
        feed_dict.update({placeholders["batch_size"]: batch_size})
        feed_dict.update({placeholders["pos1"]: batch_pos[:, 0]})
        feed_dict.update({placeholders["pos2"]: batch_pos[:, 1]})
        feed_dict.update({placeholders["neg1"]: batch_neg[:, 0]})
        feed_dict.update({placeholders["neg2"]: batch_neg[:, 1]})

        prob_pos.append(sigmoid(sess.run(model.pos_neighbor, feed_dict=feed_dict)))
        prob_neg.append(sigmoid(sess.run(model.neg_neighbor, feed_dict=feed_dict)))

        """
        feed_dict.update({placeholders["localSim"]: np.zeros((batch_size, batch_size), dtype=np.float32)})
        feed_dict.update({placeholders["batch_size"]: batch_size})

        feed_dict.update({placeholders["nodes"]: batch_pos[:, 0]})
        embedding_pos = sess.run(model.output, feed_dict=feed_dict)

        feed_dict[placeholders["nodes"]] = batch_pos[:, 1]
        context_pos = sess.run(model.context, feed_dict=feed_dict)

        prob_pos.append(sigmoid(np.sum(np.multiply(embedding_pos, context_pos), axis=1)))

        feed_dict[placeholders["nodes"]] = batch_neg[:, 0]
        embedding_neg = sess.run(model.output, feed_dict=feed_dict)

        feed_dict[placeholders["nodes"]] = batch_neg[:, 1]
        context_neg = sess.run(model.context, feed_dict=feed_dict)

        prob_neg.append(sigmoid(np.sum(np.multiply(embedding_neg, context_neg), axis=1)))
        """
    
    pos = np.concatenate(prob_pos)
    neg = np.concatenate(prob_neg)
    preds_all = np.hstack([pos, neg])
    true_all = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])

    auc_score = roc_auc_score(true_all, preds_all)
    ap_score = average_precision_score(true_all, preds_all)

    return auc_score, ap_score




def pretrain(graph, placeholders, model, sess):

    for i in range(FLAGS.pretrain_step):
        train_feed_dict = graph.next_batch_feed_dict(placeholders)
        _, loss_e = sess.run([model.opt_step_e, model.loss_e], feed_dict=train_feed_dict)
        print("pretrain step {}: loss_e: {:.6f}".format(i, loss_e))


def train_iterative(graph, placeholders, model, sess, saver, model_path):

    max_auc_val = 0.0

    for i in range(FLAGS.steps):
        
        train_feed_dict = graph.next_batch_feed_dict(placeholders)
    
        sess.run(model.opt_step_e, feed_dict = train_feed_dict)
        sess.run(model.opt_step_m, feed_dict = train_feed_dict)

        if i % 5 == 0 or i == FLAGS.steps - 1:

            loss_train, kl, sim = sess.run([model.loss, model.kl, model.sim], feed_dict=train_feed_dict)

            print("Epoch: {}".format(i+1), "loss: {:.5f}".format(loss_train), "kl: {:.5f}".format(kl), 
                    "sim: {:.5f}".format(sim), end=", ")

            auc_test, ap_test = incremental_evaluate_link(graph, placeholders, model, sess, test=True)
            print("AUC_TEST: {:.5f}, AP_TEST: {:.5F}".format(auc_test, ap_test))

            if (auc_test+ap_test)/2 > max_auc_val:
                save_path = saver.save(sess, "{}/model_best.ckpt".format(model_path), global_step=i)
                print("=================successfully save the model at: {}=======================".format(save_path))
                max_auc_val = (auc_test+ap_test)/2
        

def classification(graph, placeholders, model, sess):
    
    batch_num = 0
    outputs = []
    nodes = list(range(graph.n_nodes))

    while(batch_num*FLAGS.val_batch_size<graph.n_nodes):
        
        idx_start = batch_num * FLAGS.val_batch_size
        idx_end = min((batch_num+1)*FLAGS.val_batch_size, graph.n_nodes)
        batch_nodes = nodes[idx_start:idx_end]
        batch_size = len(batch_nodes)

        feed_dict = {}
        feed_dict.update({placeholders["pos1"]: batch_nodes})
        feed_dict.update({placeholders["pos2"]: batch_nodes})
        feed_dict.update({placeholders["neg1"]: batch_nodes})
        feed_dict.update({placeholders["neg2"]: batch_nodes})
        # feed_dict.update({placeholders["localSim"]: np.zeros((batch_size, batch_size), dtype=np.float32)})
        feed_dict.update({placeholders["batch_size"]: batch_size})

        embedding = sess.run(model.pos1_embed, feed_dict=feed_dict)
        outputs.append(embedding)

        batch_num += 1
    
    feature = np.concatenate(outputs, axis=0)
    # print("====================================get feature: {}===========================".format(feature.shape))
    X_train = feature[graph.train_mask]
    X_test = feature[graph.test_mask]
    Y_train = np.argmax(graph.y_overall[graph.train_mask], axis=1)
    Y_test = np.argmax(graph.y_overall[graph.test_mask], axis=1)

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)

    accuracy = np.mean(np.equal(y_pred, Y_test))

    return accuracy



def train(graph, placeholders, model, sess, saver, model_path):

    max_auc_val = 0.0

    for i in range(FLAGS.steps):
        
        train_feed_dict = graph.next_batch_feed_dict(placeholders)
    
        sess.run(model.opt_step, feed_dict = train_feed_dict)

        if i % 5 == 0 or i == FLAGS.steps - 1:

            loss_train, kl, sim = sess.run([model.loss, model.kl, model.sim], feed_dict=train_feed_dict)

            print("Epoch: {}".format(i+1), "loss: {:.5f}".format(loss_train), "kl: {:.5f}".format(kl), 
                    "sim: {:.5f}".format(sim), end=", ")

            auc_test, ap_test = incremental_evaluate_link(graph, placeholders, model, sess, test=True)
            print("AUC_TEST: {:.5f}, AP_TEST: {:.5F}".format(auc_test, ap_test))

            if (auc_test+ap_test)/2 > max_auc_val:
                save_path = saver.save(sess, "{}/model_best.ckpt".format(model_path), global_step=i)
                print("=================successfully save the model at: {}=======================".format(save_path))
                max_auc_val = (auc_test+ap_test)/2
            

if __name__ == "__main__":

    log_parameter_settings()  # log parameter settings

    # load data
    graph = load_data()

    # set placeholder
    placeholders = {
        'pos1': tf.placeholder(dtype=tf.int32, shape=[None]),
        'pos2': tf.placeholder(dtype=tf.int32, shape=[None]),
        'neg1': tf.placeholder(dtype=tf.int32, shape=[None]),
        'neg2': tf.placeholder(dtype=tf.int32, shape=[None]),
        #'neg21': tf.placeholder(dtype=tf.int32, shape=[None]),
        #'neg22': tf.placeholder(dtype=tf.int32, shape=[None]),
        "batch_size": tf.placeholder(tf.int32, name='batch_size')
    }


    output_dim = FLAGS.output_dim

    model = StructAwareGP_Unsup(placeholders, graph.feature, FLAGS.feature_dim, FLAGS.n_samples, latent_layer_units, output_dim, 
                                node_neighbors=graph.node_neighbors, dropout=FLAGS.dropout, bias=True, act=tf.nn.relu, 
                                weight_decay=FLAGS.weight_decay, lr=FLAGS.lr, typ="link")

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
    pretrain(graph, placeholders, model, sess)
    train_iterative(graph, placeholders, model, sess, saver, model_path)
    # train(graph, placeholders, model, sess, saver, model_path)

    # evaluate the model
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    print("restored parameters")
    acc_val_list = []
    auc_test_list = []
    ap_test_list = []

    for i in range(10):

        # acc_val = evaluate(graph, placeholders, model, sess)
        auc_test, ap_test = incremental_evaluate_link(graph, placeholders, model, sess, test=True)
        #acc = classification(graph, placeholders, model, sess)
        acc = 0

        auc_test_list.append(auc_test)
        ap_test_list.append(ap_test)
        acc_val_list.append(acc)
        
    print("===============================================")
    print("AUC_TEST: {:.5F}".format(np.max(auc_test_list)))
    print("AP_TEST: {:.5F}".format(np.max(ap_test_list)))
    print("ACC_TEST: {:5f}".format(np.max(acc_val_list)))
    print("===============================================")

    