import collections

import numpy as np
from operator import add
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from PyNomaly import loop
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import euclidean


def __first_moment_estimator(projected, t, n):
    f1 = [0] * n
    for i in range(0, t):
        cl = [0] * n
        cr = [0] * n
        li = projected[i]
        for j in range(0, n):
            idx = li[j][0]
            cl[idx] = j - 1
            cr[idx] = n - 1 - cl[idx]
        for j in range(0, n):
            f1[j] += cl[j] * cr[j]
    return list(map(lambda x: x * ((2 * math.pi) / (t * (n - 1) * (n - 2))), f1))


def __frobenius_norm(train, t, n):
    f2 = [0] * n
    sl = np.random.choice([-1, 1], size=(n,), p=[1. / 2, 1. / 2])
    sr = np.random.choice([-1, 1], size=(n,), p=[1. / 2, 1. / 2])
    for i in range(0, t):
        amsl = [0] * n
        amsr = [0] * n
        li = train[i]
        for j in range(1, n):
            idx1 = li[j][0]
            idx2 = li[j - 1][0]
            amsl[idx1] = amsl[idx2] + sl[idx2]
        for j in range(n - 2, -1, -1):
            idx1 = li[j][0]
            idx2 = li[j + 1][0]
            amsr[idx1] = amsr[idx2] + sr[idx2]
        for j in range(0, n):
            f2[j] += amsl[j] * amsr[j]
    return f2


def fast_voa(projected, n, t, s1, s2):
    f1 = __first_moment_estimator(projected, t, n)
    y = []
    for i in range(0, s2):
        s = [0] * n
        for j in range(0, s1):
            result = list(map(lambda x: x ** 2, __frobenius_norm(projected, t, n)))
            s = list(map(add, s, result))
        s = list(map(lambda x: x / s1, s))
        y.append(s)
    y = list(map(list, zip(*y)))
    f2 = []
    for i in range(0, n):
        f2.append(np.average(y[i]))
    var = [0] * n
    for i in range(0, n):
        f2[i] = (4 * (math.pi ** 2) / (t * (t - 1) * (n - 1) * (n - 2))) * f2[i] - \
                (2 * math.pi * f1[i]) / (t - 1)
        var[i] = f2[i] - (f1[i] ** 2)
    return var


def recursive(X, level, labels):
    clusters = KMeans(n_clusters=20, n_jobs=4, random_state=0).fit(X)
    predict = list()
    counter = dict()
    for label in clusters.labels_:
        if not label in counter.keys():
            counter.update({label: 0})
        counter[label] += 1
    print(counter)
    for label in range(20):
        if counter[label] < 3:
            for i in range(len(X)):
                if clusters.labels_[i] == label:
                    predict.append(i)
        elif counter[label] > 30 and level > 0:
            subspace = list()
            index = 0
            mapping = dict()
            for i in range(len(X)):
                if clusters.labels_[i] == label:
                    subspace.append(X[i])
                    mapping.update({index: i})
                    index += 1
            subspace = np.array(subspace)
            p = recursive(subspace, level-1, labels)
            for inde in p:
                predict.append(mapping[inde])
    return predict
    #     if counter[label] > max_count:
    #         max_label = label
    #         max_count = counter[label]
    # pivot_label = max_label
    # for i in range(len(X)):
    #     label = clusters.labels_[i]
    #     if label != pivot_label and counter[label] < 3:
    #         predict.append(i)
    #     if counter[label] == 768:
    #         predict.append(i)
    #     if counter[label] == 149:
    #         predict.append(i)
    # if level == 0:
    #     return predict
    # else:
    #     subspace = list()
    #     index = 0
    #     mapping = dict()
    #     for i in range(len(X)):
    #         if clusters.labels_[i] == pivot_label:
    #             subspace.append(X[i])
    #             mapping.update({index: i})
    #             index += 1
    #     subspace = np.array(subspace)
    #     p = recursive(subspace, level-1, labels)
    #     for inde in p:
    #         predict.append(mapping[inde])
    #     return predict


def k_means(S, n_clusters, outlier_cluster_size_limit):
    # level = 1
    X = np.array(S)
    predict = list()
    clusters = KMeans(n_clusters=n_clusters, n_jobs=4, random_state=0).fit(X)
    counter = dict()
    for label in clusters.labels_:
        if not label in counter.keys():
            counter.update({label: 0})
        counter[label] += 1
    for i in range(len(X)):
        label = clusters.labels_[i]
        if counter[label] < outlier_cluster_size_limit :
            predict.append(1)
        else:
            predict.append(0)
    # print(counter)

    return predict


def DB_Scan(S, eps, min_samples):
    X = np.array(S)
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    predict = clusters.labels_
    index = 0
    for i in predict:
        if i < 0:
            predict[index] = 1
        else:
            predict[index] = 0
        index += 1
    return predict


def LOF(S, n_neighbors, contamination):
    X = np.array(S)
    clusters = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(X)
    predict = clusters
    for i in range(len(predict)):
        if predict[i] == -1:
            predict[i] = 1
        else:
            predict[i] = 0
    return predict


def loOP(S, n_neighbours):
    X = np.array(S)
    m = loop.LocalOutlierProbability(X, extent=0.95, n_neighbors=n_neighbours).fit()
    scores = m.local_outlier_probabilities
    for i in scores:
        print(i)
    return scores


def SVM(S, nu):
    X = np.array(S)
    clf = OneClassSVM(kernel='linear', random_state=0, nu=nu)
    clf.fit(X)
    clusters = clf.predict(X)
    for i in range(len(clusters)):
        if clusters[i] == -1:
            clusters[i] = 1
        else:
            clusters[i] = 0
    return clusters


def isolation_forest(S, contamination):
    X = np.array(S)
    clf = IsolationForest(contamination=contamination)
    clf.fit(X)
    clusters = clf.predict(X)
    for i in range(len(clusters)):
        if clusters[i] == -1:
            clusters[i] = 1
        else:
            clusters[i] = 0
    print(clusters)
    return clusters
