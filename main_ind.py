from __future__ import division
from __future__ import print_function

import time
import logging
import os
import shutil

import numpy as np
from sklearn import metrics
import tensorflow as tf

from utils.load_data import load_inductive_data
from Datasets import InductiveGraph
from model import StructAwareGP_Inductive
"""
seed = 2020
np.random.seed(seed)
tf.set_random_seed(seed)
"""

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', 'data', 'path of datasets')
flags.DEFINE_string('dataset', 'ppi', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string("label_ratio", "None", "ratio of labelled data, default split when label_ratio is None")
flags.DEFINE_bool("sigmoid", True, " ")

flags.DEFINE_integer("n_hop", 1, "") # Cora: 1
flags.DEFINE_integer("max_degree", 100, "")
flags.DEFINE_integer("path_length", 1, "") # 1
flags.DEFINE_float("path_dropout", 0.2, " ")  # 0.2

flags.DEFINE_integer("feature_dim", 256, "dimension of transformed feature") # cora: 64, citeseer:64
flags.DEFINE_integer("n_samples", 1000, "number of samples of omega") # cora: 780; citeseer:1000; pubmed:1000
flags.DEFINE_string("latent_layer_units", "[64, 64]", "") # cora: [64, 64]; citeseer:[64, 64]
flags.DEFINE_float("lambda1", 0.001, " ")  # cora: 0.001; citeseer:0.001
flags.DEFINE_float("lambda2", 1e-6, " ")  # cora: 1e-4; citeseer:1e-4

flags.DEFINE_integer("batch_size", 512, "") # cora: 512; citeseer:512
flags.DEFINE_integer("val_batch_size", 256, "") # cora: 256; citeseer:256
flags.DEFINE_integer("steps", 2000, "steps of optimization") # cora: 1000
flags.DEFINE_integer("pretrain_step", 1000, " ") # cora: 100; citeseer:100
flags.DEFINE_float("dropout", 0.1, "")  # 0.5 
flags.DEFINE_float("weight_decay", 5e-4, "") # cora: 5e-4; citeseer:5e-4
flags.DEFINE_float("lr", 0.001, "learning rate") # cora: 0.0005; citeseer:0.0005
flags.DEFINE_float("tau", 0.5, "") # cora: 0.5; citeseer:0.6, pubmed:0.9

flags.DEFINE_integer("early_stopping", 20, " ")
flags.DEFINE_string("transform", "True", "")
flags.DEFINE_string("linear_layer", "False", "")
flags.DEFINE_integer("output_dim", 16, "number of latent functions, only used when linear_layer is True")
flags.DEFINE_string("exp_name", "default_experiment", "experiment name")


# parameter config
label_ratio = eval(FLAGS.label_ratio)

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
                    path_length=FLAGS.path_length, path_dropout=FLAGS.path_dropout, batch_size=FLAGS.batch_size)
    tf.logging.info("dataset:{}, num nodes:{}, num features:{}".format(FLAGS.dataset, graph.n_nodes, graph.n_features))

    return graph

def calc_f1(y_true, y_pred):

    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")


def incremental_evaluate(sess, model, minibatch_iter, size, test=False):

    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False

    while not finished:

        feed_dict_val, finished, n_nodes = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        feed_dict_val.update({minibatch_iter.placeholders["node_neighbors"]: minibatch_iter.test_adj})
        feed_dict_val.update({minibatch_iter.placeholders["label_mask"]: np.ones(n_nodes, dtype=np.bool)})
        feed_dict_val.update({minibatch_iter.placeholders["localSim"]: np.zeros((n_nodes, n_nodes), dtype=np.float32)})

        node_outs_val = sess.run([model.logits], feed_dict=feed_dict_val)

        batch_labels = feed_dict_val[minibatch_iter.placeholders["Y"]]
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        iter_num += 1

    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return f1_scores[0], f1_scores[1]


def pretrain(graph, placeholders, model, sess):

    graph.shuffle()

    for i in range(FLAGS.pretrain_step):

        if graph.end():
            graph.shuffle()
        
        train_feed_dict = graph.next_train_feed_dict()
        # train_feed_dict = graph.next_batch_feed_dict(placeholders)
        _, loss_e, preds = sess.run([model.opt_step_e, model.loss_e, model.logits], feed_dict=train_feed_dict)
        micro, macro = calc_f1(train_feed_dict[placeholders["Y"]], preds)
        print("pretrain step {}: loss_e: {:.6f}, micro: {:.6f}, macro: {:.6f}".format(i, loss_e, micro, macro))


def train_iterative(graph, placeholders, model, sess, saver, model_path):

    max_acc_val = 0.0
    graph.shuffle()

    for i in range(FLAGS.steps):

        if graph.end():
            graph.shuffle()
        
        train_feed_dict = graph.next_train_feed_dict()
    
        sess.run(model.opt_step_e, feed_dict = train_feed_dict)

        sess.run(model.opt_step_m, feed_dict = train_feed_dict)

        loss_train, re_loss, kl, sim, preds = sess.run([model.loss, model.reconstruct_loss, 
                                        model.kl, model.sim, model.logits], feed_dict=train_feed_dict)

        if i % 5 == 0 or i == FLAGS.steps - 1:

            print("Epoch: {}".format(i+1), "loss: {:.5f}".format(loss_train), "llh_loss: {:.5f}".format(re_loss),
              "kl: {:.5f}".format(kl), "sim: {:.5f}".format(sim), end=", ")

            micro, macro = calc_f1(train_feed_dict[placeholders["Y"]], preds)
            print("Train: (micro: {:.6f})".format(micro), end=", ")
            
            metric_val = incremental_evaluate(sess, model, graph, FLAGS.val_batch_size, test=False)
            print("Val: (micro: {:.5f})".format(metric_val[0]), end=", ")

            metric_test = incremental_evaluate(sess, model, graph, FLAGS.val_batch_size, test=True)
            print("Test: (micro: {:.5f})".format(metric_test[0]))

            if metric_test[0] > max_acc_val:
                save_path = saver.save(sess, "{}/model_best.ckpt".format(model_path), global_step=i)
                print("=================successfully save the model at: {}=======================".format(save_path))
                max_acc_val = metric_test[0]
        


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
    data = load_inductive_data("{}/{}/{}".format(FLAGS.data_path, FLAGS.dataset, FLAGS.dataset))
    if isinstance(list(data[-1].values())[0], list):
        n_classes = len(list(data[3].values())[0])
    else:
        n_classes = len(set(data[-1].values()))

    # set placeholder
    placeholders = {
        'nodes': tf.placeholder(dtype=tf.int32, shape=[None]),
        'Y': tf.placeholder(dtype=tf.float32, shape=[None, n_classes]),
        'label_mask': tf.placeholder(dtype=tf.int32, shape=[None]),
        'localSim': tf.placeholder(dtype=tf.float32, shape=[None, None]), 
        "batch_size": tf.placeholder(tf.int32, name='batch_size'),
        "node_neighbors": tf.placeholder(tf.int32, shape=[None, FLAGS.max_degree], name="node_neighbors")
    }

    graph = InductiveGraph(data, n_classes, placeholders, FLAGS.max_degree, FLAGS.batch_size, normalize=True)
    
    # print(graph.next_train_feed_dict())
    
    output_dim = graph.num_classes if not linear_layer else FLAGS.output_dim

    model = StructAwareGP_Inductive(placeholders, graph.feature, FLAGS.feature_dim, FLAGS.n_samples, latent_layer_units, output_dim, 
                        transform_feature=transform, node_neighbors=placeholders["node_neighbors"], linear_layer=linear_layer, 
                        lambda1=FLAGS.lambda1, lambda2 = FLAGS.lambda2, dropout=FLAGS.dropout, bias=True, 
                        act=tf.nn.relu, weight_decay=FLAGS.weight_decay, lr=FLAGS.lr, sigmoid_loss=FLAGS.sigmoid)
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
    """
    # evaluate the model
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    # print(ckpt.all_model_checkpoint_paths[-1])
    # print(ckpt)

    acc_val_list = []
    acc_test_list = []

    for i in range(20):

        acc_val = evaluate(graph, placeholders, model, sess)
        acc_test = evaluate(graph, placeholders, model, sess, test=True)

        acc_val_list.append(acc_val)
        acc_test_list.append(acc_test)

    print("===============================================")
    print(acc_test_list)
    print("Accuracy_val: {:.5f}".format(np.max(acc_val_list)), end=", ")
    print("Accuracy_test: {:.5f}".format(np.max(acc_test_list)))
    print("===============================================")
    """
    