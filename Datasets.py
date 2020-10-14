# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:59:12 2020

@author: fangjy
"""
import os
import pandas
import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfTransformer
from utils.load_data import load_data, load_AN
from utils.util import calculate_SE_kernel, normalize_adj, preprocess_features

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

    def __init__(self, dataset_str, path="data", with_feature=True, label_ratio=None,
                n_hop=2, max_degree=128, path_length=1, path_dropout=0.5, batch_size=512):
        """
        :param dataset_str:
        :param path:
        :param exp_type: experiment type: link (link prediction) or classification
        """

        if dataset_str in ['cora', 'citeseer', 'pubmed'] and label_ratio is None:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_data(dataset_str, path)
        else:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_AN(dataset_str, path, ratio=label_ratio)

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
        self.nodes_unlabel = np.array([i for i in range(self.n_nodes) if not train_mask[i]])
        self.batch_num = 0
        self.val_batch_num = 0
        batch_size = self.n_nodes if batch_size == -1 else batch_size
        self.batch_size = batch_size - len(self.idx_train) # 每个batch都加入训练数据


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


# {5: [1629, 2546, 1659], 1629: [1711, 1659, 5], 2546: [952, 5, 466, 628], 1659: [1629, 5]}
if __name__ == "__main__":
    # from Datasets import Graph
    g = Graph("citeseer", n_hop=1, max_degree=128)
    # subgraph = g.get_subgraph(5, 2, 128)
    # print(subgraph)
