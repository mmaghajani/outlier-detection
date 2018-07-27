import numpy as np
from operator import add
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

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


def k_means(S):
    X = np.array(S)
    clusters = KMeans(n_clusters=2, random_state=0).fit(X)
    return clusters.labels_


def DB_Scan(S, eps, min_samples):
    X = np.array(S)
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return clusters.labels_