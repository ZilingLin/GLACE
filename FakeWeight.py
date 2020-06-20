# -*- coding: UTF-8 -*- 
import numpy as np
import networkx as nx
import node2vec
import random

def generate_graph_PageRank_matrix(PageRank_matrix, path, window_size):
    for k in range(len(path)):                                      #对任意walk（游走序列）
        for i in range(len(path[k])):                               #中的任意结点i。
            for j in range(i - window_size, i + window_size + 1):   #若 i 和 j 指向的结点
                if i == j or j < 0 or j >= len(path[k]):            #在同一窗口下，则加入all_pairs队列中
                    continue
                else:
                    PageRank_matrix[path[k][i]][path[k][j]] += 1

    s = np.sum(PageRank_matrix, axis=1)
    # Attention here
    s[s == 0] = 1
    PageRank_matrix /= s[:, None]

    return PageRank_matrix

def make_graph_M(A):
    G = nx.from_scipy_sparse_matrix(A)
    N = len(list(G.nodes()))
    # build adj matrix
    M = np.zeros([N, N], dtype = np.float)
    # Perform random walks to generate graph context
    node2vec_G = node2vec.Graph(G, False, 1, 1) # FLAGS.directed, FLAGS.p, FLAGS.q)
    node2vec_G.preprocess_transition_probs()
    walks = node2vec_G.simulate_walks(10, 80) # (FLAGS.num_walks, FLAGS.walk_length)
    M = generate_graph_PageRank_matrix(M, walks, 10) # window_size

    return M



def construct_traget_neighbors(nx_G, X, FLAGS):
    # construct target neighbor feature matrix
    mode = FLAGS.target_neighbors
    X_target = np.zeros(X.shape)
    nodes = nx_G.nodes()

    if mode == 'OWN':
        # autoencoder for reconstructing itself
        return X
    elif mode == 'EMN':
        # autoencoder for reconstructing Elementwise Median Neighbor
        for node in nodes:
            neighbors = list(nx_G.neighbors(node))
            if len(neighbors) == 0:
                X_target[node] = X[node]
            else:
                temp = np.array(X[node])
                for n in neighbors:
                    if FLAGS.weighted:
                        # weighted sum
                        temp = np.vstack((temp, X[n] * nx_G[node][n]['weight']))
                    else:
                        temp = np.vstack((temp, X[n]))
                temp = np.median(temp, axis=0)
                X_target[node] = temp
        return X_target
    elif mode == 'WAN':
        # autoencoder for reconstructing Weighted Average Neighbor
        for node in nodes:
            # put myself ( maybe can be deleted, because second-order will count itself)
            temp = np.array(X[node])
            # a = temp
            # total = len(nodes)
            # weights = np.array(1)

            # compute neighbors
            alpha = 0.5
            # beta = 0
            first_order = 1
            neighbors = list(nx_G.neighbors(node))
            neighbors_length = len(neighbors)
            if neighbors_length != 0:
                for n in neighbors:
                    if FLAGS.weighted:
                        # weighted sum
                        temp = np.vstack((temp, X[n] * nx_G[node][n]['weight']))
                    else:
                        temp = np.vstack((temp, X[n]))
                        set_n = set(nx_G.neighbors(n))
                        set_n = set_n.intersection(set(neighbors))
                        second_order = len(set_n) / neighbors_length
                        # weights = np.append(weights, first_order)
                        w = alpha * first_order + (1 - alpha) * second_order
                        # nx_G[node][n]['weight'] = w
                        # b = np.array(X[n])
                        # w = np.dot(a, b)
                        # nx_G[node][n]['weight'] = total / (w + 1)
                temp = np.mean(temp, axis=0)
                # temp = np.average(temp, axis=0, weights = weights)
                X_target[node] = temp
        return X_target


def make_small_graph(G, ratio):
    # ratio is the test set size divide by the whole graph
    test_size = int(G.number_of_edges() * ratio)

    edge_list=list(G.edges)
    np.random.shuffle(edge_list)
    test_set = edge_list[:test_size]
    train_set = edge_list[test_size:]

    non_edge_list=list(nx.non_edges(G))    
    np.random.shuffle(non_edge_list)
    test_negative_set = non_edge_list[:test_size]

    return train_set, test_set, test_negative_set


