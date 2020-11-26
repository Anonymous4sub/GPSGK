# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:59:12 2020

@author: fangjy
"""
import os
import time
import pandas
import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfTransformer
from utils.load_data import load_data, load_AN, load_amazon_ca
from utils.util import calculate_SE_kernel, normalize_adj, preprocess_features, sparse_to_tuple

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import csv


class ShortestPathAttr(object):

    def __init__(self, normalize=True, metric=np.dot, **kwargs):

        self.normalize = normalize
        self.metric = metric

    def compare(self, sp1, sp2, feature1, feature2):

        kernel = 0
        len_s1 = sp1.shape[0]
        len_s2 = sp2.shape[0]

        for i in range(len_s1):
            for j in range(len_s1):
                if j == i :
                    continue
                for m in range(len_s2):
                    for k in range(len_s2):
                        if k == m:
                            continue 
                        if (sp1[i, j] == sp2[m, k] and sp1[i, j] != float("Inf")):
                            kernel += self.metric(feature1[i], feature2[m]) * self.metric(feature1[j], feature2[k])
        return kernel
    

    def compare_v2(self, sp1, sp2, fea1, fea2):

        n1 = sp1.shape[0]
        n2 = sp2.shape[0]

        metrics = np.zeros((n1, n2), dtype=np.float32)
        for i in range(n1):
            for j in range(n2):
                metrics[i, j] = self.metric(fea1[i], fea2[j])
        
        kernels = np.multiply(np.expand_dims(metrics, (1, -1)), np.expand_dims(metrics, (0, -2)))
        # print(kernels.shape)

        for i in range(n1):
            sp1[i, i] = float("inf")
        for i in range(n2):
            sp2[i, i] = float("inf")

        sp1 = np.expand_dims(sp1, (-1, -2))
        sp2 = np.expand_dims(sp2, (0, 1))
        mask = (np.equal(sp1, sp2)) & (np.not_equal(sp1, float("inf")) & np.not_equal(sp2, float("inf")))
        print(mask.shape)

        kernel = np.sum(kernels * mask)

        return kernel

    
    def compare_list(self, subgraphs, feature):

        """
        compare a list of subgraphs

        subgraphs: a list of subgraphs in the dict format
        feature: a numpy array of all node feature
        """
        n = len(subgraphs)

        shortestpaths = []
        for i in range(n):
            g = nx.Graph(subgraphs[i])
            shortestpaths.append(nx.floyd_warshall_numpy(g))
        
        globalSim = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i+1):
                sim = self.compare_v2(shortestpaths[i], shortestpaths[j], feature[list(subgraphs[i].keys())], feature[list(subgraphs[j].keys())])
                globalSim[i][j] = sim
                globalSim[j][i] = sim
        
        if self.normalize:
            d = np.diag(globalSim)
            norm = np.sqrt(np.outer(d, d))
            globalSim /= (norm + 1e-6)
        
        return globalSim


class Graph(object):

    def __init__(self, dataset_str, path="data", with_feature=True, label_ratio=None, n_hop=2, 
                max_degree=128, path_length=1, path_dropout=0.5, batch_size=512, small=True):
        """
        :param dataset_str:
        :param path:
        :param exp_type: experiment type: link (link prediction) or classification
        """

        if dataset_str in ['cora', 'citeseer', 'pubmed'] and label_ratio is None:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_data(dataset_str, path)
        elif dataset_str in ['cora', 'citeseer', 'pubmed']:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_AN(dataset_str, path, ratio=label_ratio)
        else:
            # label_ratio = 0.01
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_amazon_ca(dataset_str, path, label_ratio=label_ratio)

        self.n_nodes, self.n_features = features.shape
        self.n_classes = y_train.shape[1]

        self.graph = graph
        self.adj = normalize_adj(adj).astype(np.float32)
        self.node_neighbors, self.node_degrees = self.get_neighbors_degree(max_degree) # node_neighbors:[n_nodes+1, max_degree]

        if not with_feature:
            features = np.eye(features.shape[0], dtype=features.dtype)
        # self.feature = self.get_tfidf_feature(features, dataset_str, normalize_feature)  # numpy array
        feature = preprocess_features(features)
        self.feature = np.vstack([feature, np.zeros((self.n_features,))]).astype(np.float32) # add a dummy node for feature aggregation

        # self.y_train = y_train[train_mask]  # n_y_train, n_classes
        # self.y_val = y_val[val_mask]
        # self.y_test = y_test[test_mask]
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.y_overall = self.y_train + self.y_val + self.y_test
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.idx_train = np.arange(self.n_nodes)[train_mask]
        self.idx_val = np.arange(self.n_nodes)[val_mask]
        self.idx_test = np.arange(self.n_nodes)[test_mask]
        
        # get global similarity
        # self.globalSim = self.load_globalSim(dataset_str, n_hop, max_degree, path_length, path_dropout)
        
        # settings for batch sampling
        self.nodes_label = np.array([i for i in range(self.n_nodes) if train_mask[i]])
        self.nodes_unlabel = np.array([i for i in range(self.n_nodes) if not train_mask[i]])
        self.val_batch_num = 0
        if small:
            print("====================================batch with small size of training data=====================================")
            self.batch_num = 0
            batch_size = self.n_nodes if batch_size == -1 else batch_size
            self.batch_size = batch_size - len(self.idx_train) # 每个batch都加入训练数据
        else:
            print("====================================batch with large size of training data=====================================")
            self.batch_num_label = 0
            self.batch_num_unlabel = 0
            self.batch_size_label = int(np.round(batch_size * 0.5))
            self.batch_size_unlabel = int(np.round(batch_size * 0.5))


    def val_batch_feed_dict(self, placeholders, val_batch_size=256, test=False, localSim=True):

        if test:
            nodes = self.idx_test
        else:
            nodes = self.idx_val

        if self.val_batch_num * val_batch_size >= len(nodes):
            self.val_batch_num = 0
            return None
        
        idx_start = self.val_batch_num * val_batch_size
        idx_end = min(idx_start + val_batch_size, len(nodes))
        self.val_batch_num += 1

        batch_nodes = nodes[idx_start:idx_end]        
        
        
        feed_dict = {}
        n_nodes = len(batch_nodes)
        feed_dict.update({placeholders["nodes"]: batch_nodes})
        feed_dict.update({placeholders['Y']: self.y_overall[batch_nodes]})
        feed_dict.update({placeholders['label_mask']: np.ones(n_nodes, dtype=np.bool)})
        if localSim:
            feed_dict.update({placeholders['localSim']: self.adj[batch_nodes][:, batch_nodes]})
        feed_dict.update({placeholders["batch_size"]: n_nodes})

        return feed_dict
        

    def next_batch_feed_dict(self, placeholders, localSim=True):

        if self.batch_num * self.batch_size >= len(self.nodes_unlabel):
            self.nodes_unlabel = np.random.permutation(self.nodes_unlabel)
            self.batch_num = 0
        
        idx_start = self.batch_num * self.batch_size
        idx_end = min(idx_start + self.batch_size, len(self.nodes_unlabel))
        self.batch_num += 1

        batch_unlabel = self.nodes_unlabel[idx_start : idx_end]
        batch_nodes = np.concatenate([self.idx_train, batch_unlabel])

        return self.batch_feed_dict(batch_nodes, placeholders, localSim)

    
    def next_batch_feed_dict_v2(self, placeholders, localSim=True):

        """
        适用于有标签的数据比较多的情况；
        """
        if self.batch_num_label * self.batch_size_label >= len(self.nodes_label):
            self.nodes_label = np.random.permutation(self.nodes_label)
            self.batch_num_label = 0
        if self.batch_num_unlabel * self.batch_size_unlabel >= len(self.nodes_unlabel):
            self.nodes_unlabel = np.random.permutation(self.nodes_unlabel)
            self.batch_num_unlabel = 0
        
        idx_start_label = self.batch_num_label * self.batch_size_label 
        idx_end_label = min(idx_start_label + self.batch_size_label, len(self.nodes_label))
        self.batch_num_label += 1 
        idx_start_unlabel = self.batch_num_unlabel * self.batch_size_unlabel 
        idx_end_unlabel = min(idx_start_unlabel + self.batch_size_unlabel, len(self.nodes_unlabel))
        self.batch_num_unlabel += 1

        batch_label = self.nodes_label[idx_start_label:idx_end_label]
        batch_unlabel = self.nodes_unlabel[idx_start_unlabel:idx_end_unlabel]
        batch_nodes = np.concatenate([batch_label, batch_unlabel])

        return self.batch_feed_dict(batch_nodes, placeholders, localSim=localSim) 


    def batch_feed_dict(self, nodes, placeholders, localSim):

        feed_dict = {}
        
        feed_dict.update({placeholders["nodes"]: nodes})
        feed_dict.update({placeholders['Y']: self.y_train[nodes]})
        feed_dict.update({placeholders['label_mask']: self.train_mask[nodes]})
        if localSim:
            feed_dict.update({placeholders['localSim']: self.adj[nodes][:, nodes]})
        # feed_dict.update({placeholders["globalSim"]: self.globalSim[nodes][:, nodes]})
        feed_dict.update({placeholders["batch_size"]: len(nodes)})
        
        return feed_dict
        # return nodes, self.y_train[nodes], self.train_mask[nodes], self.adj[nodes][:, nodes], self.globalSim[nodes][:, nodes]

    

    def load_globalSim(self, dataset_str, n_hop, max_degree, path_length, path_dropout):

        if not os.path.exists('data/{}/globalSim'.format(dataset_str)):
            os.mkdir('data/{}/globalSim'.format(dataset_str))
    
        globalSim_file = "data/{}/globalSim/hop{}_max{}.npy".format(dataset_str, n_hop, max_degree)
        if not os.path.exists(globalSim_file):
            globalSim = self.get_globalSim_v2(n_hop, max_degree, path_length, path_dropout)
            np.save(globalSim_file, globalSim)
            print("=====successfully save globaSim file of {} dataset=====".format(dataset_str))
            print("=====maximum in globalSim:{}, minimum in globalSim: {}".format(np.max(globalSim), np.min(globalSim)))
        else:
            globalSim = np.load(globalSim_file)

        return globalSim
        

    def get_neighbors_degree(self, max_degree):

        adj = self.n_nodes * np.ones((self.n_nodes + 1, max_degree), dtype=np.int32)
        deg = np.ones(self.n_nodes, dtype=np.int32)

        for node in range(self.n_nodes):
            neighbors = np.array(self.graph[node])
            deg[node] = len(neighbors)
            if deg[node] == 0:
                continue
            if deg[node] > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
            elif deg[node] < max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=True)

            adj[node, :] = neighbors

        return adj, deg


    def get_tfidf_feature(self, feature, dataset_str, normalize=False):

        if sp.issparse(feature):
            feature = feature.todense()

        if dataset_str != "pubmed":
            transformer = TfidfTransformer(smooth_idf=True)
            feature = transformer.fit_transform(feature).toarray()

        if normalize:
            norm = np.linalg.norm(feature, axis=1, keepdims=True)
            feature = feature / np.where(norm == 0., 1, norm)
        
        feature = feature.astype(np.float32)

        return feature


    def normalize_matrix(self, matrix):

        return matrix / np.sqrt(np.sum(np.square(matrix)))

    def get_globalSim_v2(self, n_hop, max_degree, path_length, path_dropout=0.0):

        subgraphs = []
        for i in range(self.n_nodes):
            subgraphs.append(self.get_subgraph_v2(i, n_hop, max_degree))
        print("successfully obtain subgraphs")
        
        spkerl = ShortestPathAttr(normalize=True, metric=np.dot)
        globalSim = spkerl.compare_list(subgraphs, self.feature)

        return globalSim


    def get_globalSim(self, n_hop, max_degree, path_length, path_dropout=0.0):

        globalSim = np.eye(self.n_nodes, dtype=np.float32)

        for i in range(self.n_nodes):

            if i % 200 == 0:
                print("get globalSim befor node {}".format(i))

            for j in range(i):

                subgraph_i = self.get_subgraph_v2(i, n_hop, max_degree)
                subgraph_j = self.get_subgraph_v2(j, n_hop, max_degree)
                # print("size of subgraph_{} and subgraph{}: {}, {}".format(i, j, len(subgraph_i), len(subgraph_j)))

                paths_i = self.get_paths_feature(subgraph_i, path_length, path_dropout)  # n_path, n_feature*(n_hop+1)
                paths_j = self.get_paths_feature(subgraph_j, path_length, path_dropout)

                subgraph_sim = calculate_SE_kernel(paths_i, paths_j)  # n_path_i, n_path_j
                subgraph_sim = np.mean(subgraph_sim)

                globalSim[i, j] = subgraph_sim
                globalSim[j, i] = subgraph_sim
        
        # globalSim = self.normalize_matrix(globalSim)

        return globalSim
    

    def get_subgraph(self, node_i, n_hop, max_degree):

        subgraph = defaultdict(list)
        hops = [deque([node_i])]
        for _ in range(n_hop - 1):
            hops.append(deque())
        # print(hops)
        for i in range(n_hop):
            
            nodes = hops[i]

            while(len(nodes)>0):
                node = nodes.popleft()
                neighbors = self.get_neighbors(node, max_degree)

                subgraph[node].extend(neighbors)
            
                if (i+1) < n_hop:
                    for n in neighbors:
                        if n not in subgraph.keys():    # n没有被遍历过
                            hops[i+1].append(n)
        
        return subgraph
    
    
    def get_subgraph_v2(self,node_i, n_hop, max_degree):
        
        subgraph = defaultdict(list)
        
        current = deque([node_i])
        tranversed = set()
        
        
        for _ in range(n_hop):
            
            next_hop = deque()

            while(len(current)>0):
                node = current.popleft()
                if node in tranversed:
                    continue
                neighbors = self.get_neighbors(node, max_degree)
                
                for n in neighbors:
                    if n not in subgraph[node]:
                        subgraph[node].append(n)
                    if node not in subgraph[n]:
                        subgraph[n].append(node)
                    next_hop.append(n)
                
                tranversed.add(node)
            
            current = next_hop
                   
        return subgraph
                

    
    def get_neighbors(self, node, max_degree):

        if len(self.graph[node]) <= max_degree:
            return self.graph[node]

        else:
            neighbors = np.random.permutation(self.graph[node])[:max_degree]
            return list(neighbors)

    
    def get_paths_feature(self, subgraph, path_length, path_dropout):

        # obtain paths of length path_length and their features
        paths = []
        nodes = list(subgraph.keys())
        for node in nodes:
            node_paths = node_start_path(subgraph, node, path_length)
            if len(node_paths) > 0:
                paths.extend(node_paths)

        # 单独一个节点的子图时
        if len(paths) < 1:
            node_feature = self.feature[nodes[0]].copy()
            if len(node_feature.shape)<2:
                node_feature = np.expand_dims(node_feature, axis=0)
            return node_feature

        paths = np.asarray(paths, dtype=np.int32)
        n_paths = paths.shape[0]
        if n_paths > 10:
            path_filter = paths[np.random.permutation(n_paths)[: int(n_paths*(1-path_dropout))]]
        else:
            path_filter = paths

        feature = self.feature[path_filter[:, 0]]
        for i in range(path_length):
            # feature = np.concatenate((feature, self.feature[path_filter[:, i+1]]), axis=1)
            feature += self.feature[path_filter[:, i+1]]
        
        feature /= n_paths

        return feature
        

def node_start_path(subgraph, node, path_length):

    if path_length == 0:
        return [[node]]
    
    result = []
    for neighbor in subgraph[node]:
        ps = node_start_path(subgraph, neighbor, path_length-1)
        # print(ps)
        for p in ps:
            if node in p:continue
            p.append(node)
            result.append(p.copy())

    # print(result)
    return result


class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx, placeholders, label_map, num_classes, batch_size=100, max_degree=25, **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes

        self.neighbors_train = []
        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
        print("==========================={} =========================".format(len(self.train_nodes)))
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]
        print(self.adj)
        print(self.deg)
        # print(self.train_nodes)
        print("length of train_nodes: ====================== {}==================".format(len(self.train_nodes)))

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),), dtype=np.int32)

        for nodeid in self.G.nodes():
            if self.G.nodes[nodeid]['test'] or self.G.nodes[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            self.neighbors_train.append(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        batch1id = batch_nodes
        batch1 = [self.id2idx[n] for n in batch1id]

        labels = np.vstack([self._make_label_vec(node) for node in batch1id])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['nodes']: batch1})
        feed_dict.update({self.placeholders['Y']: labels})

        # return feed_dict, labels
        # return feed_dict
        return batch1, feed_dict

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):

        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes

        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, 
            len(val_nodes))]

        # add a dummy neighbor
        _, feed_dict_val = self.batch_feed_dict(val_node_subset)
        # return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset
        return feed_dict_val, (iter_num+1)*size >= len(val_nodes), len(val_node_subset)

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0



class InductiveGraph(NodeMinibatchIterator):

    def __init__(self, data, num_classes, placeholder, max_degree, batch_size, normalize=True):
        
        """
        placeholders = {
        'nodes': tf.placeholder(dtype=tf.int32, shape=[None]),
        'Y': tf.placeholder(dtype=tf.float32, shape=[None, len(list(data[3].values())[0])]),
        'localSim': tf.placeholder(dtype=tf.float32, shape=[None, None]), 
        "batch_size": tf.placeholder(tf.int32, name='batch_size')
        }
        """
        
        super(InductiveGraph, self).__init__(data[0], data[2], placeholder, data[3], num_classes, 
                                            batch_size=batch_size, max_degree=max_degree)
        

        self.feature = data[1].astype(np.float32)

        self.n_nodes = len(self.G.nodes())
        self.n_features = self.feature.shape[1]

        if not self.feature is None:
            # pad with dummy zero vector
            self.feature = np.vstack([self.feature, np.zeros((self.n_features,), dtype=np.float32)])



    def next_train_feed_dict(self):

        batch_nodes, feed_dict = super().next_minibatch_feed_dict() # update batch_size, batch, Y

        t = time.time()
        # update localSim
        minibatch_size = len(batch_nodes)

        node_id_map = dict()
        for (i, node) in zip(range(minibatch_size), batch_nodes):
            node_id_map[node] = i 

        localSim = np.zeros(shape=(minibatch_size, minibatch_size), dtype=np.float32)
        # print(batch_nodes)

        for node in batch_nodes:
            #for neighbor in np.unique(self.adj[node]):
            for neighbor in self.neighbors_train[node]:
                if neighbor not in batch_nodes:
                    continue
                localSim[node_id_map[node], node_id_map[neighbor]] = 1.0

        feed_dict.update({self.placeholders["localSim"]: localSim})
        feed_dict.update({self.placeholders["label_mask"]: np.ones(minibatch_size, dtype=np.bool)})
        feed_dict.update({self.placeholders["node_neighbors"]: self.adj})

        # print("==============={}==============".format(time.time()-t))

        return feed_dict


class LinkGraph(object):

    def __init__(self, dataset_str, path="data", with_feature=True, label_ratio=None, max_degree=128, batch_size=512):

        if dataset_str in ['cora', 'citeseer', 'pubmed'] and label_ratio is None:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_data(dataset_str, path)
        elif dataset_str in ['BlogCatalog', 'Flickr', 'cora', 'citeseer', 'pubmed']:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_AN(dataset_str, path, ratio=label_ratio)
        else:
            #label_ratio = 0.2
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_amazon_ca(dataset_str, path, label_ratio=label_ratio)

        self.n_nodes, self.n_features = features.shape
        self.n_classes = y_train.shape[1]

        # split edges into training, validation and test sets
        #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, graph = self.make_train_test_edges(adj)
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, graph = self.mask_test_edges(adj)
        
        self.adj = adj_train.toarray()
        self.train_edges = train_edges
        self.val_edges = val_edges
        self.val_edges_false = val_edges_false
        self.test_edges = test_edges
        self.test_edges_false = test_edges_false

        self.graph = graph
        self.node_neighbors, self.node_degrees = self.get_neighbors_degree(max_degree) # node_neighbors:[n_nodes+1, max_degree]

        if not with_feature:
            features = np.eye(features.shape[0], dtype=features.dtype)
        feature = preprocess_features(features)
        self.feature = np.vstack([feature, np.zeros((self.n_features,))]).astype(np.float32) # add a dummy node for feature aggregation

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.y_overall = self.y_train + self.y_val + self.y_test
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.idx_train = np.arange(self.n_nodes)[train_mask]
        self.idx_val = np.arange(self.n_nodes)[val_mask]
        self.idx_test = np.arange(self.n_nodes)[test_mask]

        # settings for batch sampling
        self.batch_num = 0
        self.batch_size = batch_size
        self.val_batch_num = 0


    def val_batch_feed_dict(self, placeholders, val_batch_size=256, test=False, localSim=True):

        if test:
            nodes = self.idx_test
        else:
            nodes = self.idx_val

        if self.val_batch_num * val_batch_size >= len(nodes):
            self.val_batch_num = 0
            return None
        
        idx_start = self.val_batch_num * val_batch_size
        idx_end = min(idx_start + val_batch_size, len(nodes))
        self.val_batch_num += 1

        batch_nodes = nodes[idx_start:idx_end]        
        
        feed_dict = {}
        n_nodes = len(batch_nodes)
        feed_dict.update({placeholders["nodes"]: batch_nodes})
        feed_dict.update({placeholders['Y']: self.y_overall[batch_nodes]})
        feed_dict.update({placeholders['label_mask']: np.ones(n_nodes, dtype=np.bool)})
        if localSim:
            feed_dict.update({placeholders['localSim']: self.adj[batch_nodes][:, batch_nodes]})
        feed_dict.update({placeholders["batch_size"]: n_nodes})

        return feed_dict
        

    def next_batch_feed_dict(self, placeholders):

        if self.batch_num * self.batch_size >= len(self.train_edges):
            self.train_edges = np.random.permutation(self.train_edges)
            self.batch_num = 0
        
        idx_start = self.batch_num * self.batch_size
        idx_end = min(idx_start + self.batch_size, len(self.train_edges))
        self.batch_num += 1

        batch_edges = self.train_edges[idx_start : idx_end]
        batch_pos1_edges = []
        batch_pos2_edges = []
        batch_neg1_edges = []
        batch_neg2_edges = []

        #k = 2
        for edge in batch_edges:
            k = 1
            batch_pos1_edges.append(edge[0])
            batch_pos2_edges.append(edge[1])
            while k > 0:
                j = np.random.randint(0, self.n_nodes)
                while j in self.graph[edge[0]]:
                    j = np.random.randint(0, self.n_nodes)
                
                batch_neg1_edges.append(edge[0])
                batch_neg2_edges.append(j)
                k = k - 1
        
        """
        for _ in range(2*len(batch_edges)):
            i = np.random.randint(0, self.n_nodes)
            j = np.random.randint(0, self.n_nodes)

            while j in self.graph[i]:
                i = np.random.randint(0, self.n_nodes)
                j = np.random.randint(0, self.n_nodes)

            batch_neg1_edges.append(i)
            batch_neg2_edges.append(j)
        """

        data = (batch_pos1_edges, batch_pos2_edges, batch_neg1_edges, batch_neg2_edges)

        return self.batch_feed_dict(data, placeholders)


    def batch_feed_dict(self, data, placeholders):

        batch_pos1_edges, batch_pos2_edges, batch_neg1_edges, batch_neg2_edges = data 

        feed_dict = {}
        feed_dict.update({placeholders["pos1"]: batch_pos1_edges})
        feed_dict.update({placeholders["pos2"]: batch_pos2_edges})
        feed_dict.update({placeholders["neg1"]: batch_neg1_edges[:len(batch_pos1_edges)]})
        feed_dict.update({placeholders["neg2"]: batch_neg2_edges[:len(batch_pos2_edges)]})
        #feed_dict.update({placeholders["neg21"]: batch_neg1_edges[len(batch_pos1_edges):]})
        #feed_dict.update({placeholders["neg22"]: batch_neg2_edges[len(batch_pos2_edges):]})
        feed_dict.update({placeholders["batch_size"]: len(batch_pos1_edges)})
        
        return feed_dict
        

    def get_neighbors_degree(self, max_degree):

        adj = self.n_nodes * np.ones((self.n_nodes + 1, max_degree), dtype=np.int32)
        deg = np.ones(self.n_nodes, dtype=np.int32)

        for node in range(self.n_nodes):
            neighbors = np.array(self.graph[node])
            deg[node] = len(neighbors)
            if deg[node] == 0:
                continue
            if deg[node] > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
            elif deg[node] < max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=True)

            adj[node, :] = neighbors

        return adj, deg

    
    def get_neighbors(self, node, max_degree):

        if len(self.graph[node]) <= max_degree:
            return self.graph[node]

        else:
            neighbors = np.random.permutation(self.graph[node])[:max_degree]
            return list(neighbors)


    def make_train_test_edges(self, adj, p_val=0.05, p_test=0.10):
        """
        adj is an adjacant matrix (scipy sparse matrix)

        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
        adj_train : training adjacant matrix
        train_edges : array indicating the training edges
        val_edges : array indicating the validation edges
        val_edge_false: array indicating the false edges in validation dataset
        """
        print("================================= randomly split edges ============================")

        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()

        adj_row = adj.nonzero()[0]
        adj_col = adj.nonzero()[1]

        # get deges from adjacant matrix
        edges = []
        edges_dic = {}
        for i in range(len(adj_row)):
            edges.append([adj_row[i], adj_col[i]])
            edges_dic[(adj_row[i], adj_col[i])] = 1

        # split the dataset into training,validation and test dataset
        num_test = int(np.floor(len(edges) * p_test))
        num_val = int(np.floor(len(edges) * p_val))
        all_edge_idx = np.arange(len(edges))
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        train_edge_idx = all_edge_idx[(num_val + num_test):]

        edges = np.asarray(edges)
        test_edges = edges[test_edge_idx]  # numpy array
        val_edges = edges[val_edge_idx]  # numpy array
        train_edges = edges[train_edge_idx]  # numpy array

        test_edges_false = []
        val_edges_false = []
        false_edges_dic = {}
        while len(test_edges_false) < num_test or len(val_edges_false) < num_val:
            i = np.random.randint(0, adj.shape[0])
            j = np.random.randint(0, adj.shape[0])
            if (i, j) in edges_dic:
                continue
            if (j, i) in edges_dic:
                continue
            if (i, j) in false_edges_dic:
                continue
            if (j, i) in false_edges_dic:
                continue
            else:
                false_edges_dic[(i, j)] = 1
                false_edges_dic[(j, i)] = 1

            if np.random.random_sample() > 0.333:
                if len(test_edges_false) < num_test:
                    test_edges_false.append([i, j])
                else:
                    if len(val_edges_false) < num_val:
                        val_edges_false.append([i, j])
            else:
                if len(val_edges_false) < num_val:
                    val_edges_false.append([i, j])
                else:
                    if len(test_edges_false) < num_test:
                        test_edges_false.append([i, j])

        val_edges_false = np.asarray(val_edges_false)
        test_edges_false = np.asarray(test_edges_false)
        data = np.ones(train_edges.shape[0], dtype=adj.dtype)
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T

        # build train graph
        graph = defaultdict(list)
        row, col = adj_train.nonzero()
        for (i, j) in zip(row, col):
            graph[i].append(j)

        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, graph


    def mask_test_edges(self, adj, p_val=0.05, p_test=0.1):
        # Function to build test set with 10% positive links

        print("============================== mask test edges =======================")
        
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_test = int(np.floor(edges.shape[0] * p_test))
        num_val = int(np.floor(edges.shape[0] * p_val))

        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

        assert ~ismember(test_edges_false, edges_all)
        assert ~ismember(val_edges_false, edges_all)
        assert ~ismember(val_edges, train_edges)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)

        train_edges = np.asarray(train_edges)
        val_edges = np.asarray(val_edges)
        val_edges_false = np.asarray(val_edges_false)
        test_edges = np.asarray(test_edges)
        test_edges_false = np.asarray(test_edges_false)

        data = np.ones(train_edges.shape[0])
        # Re-build adj matrix
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T

        # build train graph
        graph = defaultdict(list)
        row, col = adj_train.nonzero()
        for (i, j) in zip(row, col):
            graph[i].append(j)

        # NOTE: these edge lists only contain single direction of edge!
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, graph


# {5: [1629, 2546, 1659], 1629: [1711, 1659, 5], 2546: [952, 5, 466, 628], 1659: [1629, 5]}
# if __name__ == "__main__":
    # from Datasets import Graph
    # g = Graph("citeseer", n_hop=1, max_degree=128)
    # subgraph = g.get_subgraph(5, 2, 128)
    # print(subgraph)
