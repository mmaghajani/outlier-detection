import random
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def random_projection(S, t):
    """
    Projects data space to random vector space
    :param S: Train set
    :param t: Amount of final dimension
    :return: A list L = L1L2L3...Lt where Li is a list of points
            ordered by their dot product with ri
    """
    l = []
    for i in range(0, t):
        ri = []
        for j in range(0, S.shape[1]):
            ri.append(random.randint(0, 1))
        l.append([])
        for index, record in S.iterrows():
            dotted = np.dot(record, ri)
            l[i].append((index, dotted))
        l[i] = sorted(l[i], key=lambda x: x[1])
    return l


def prepare_projected_data(projected, t):
    result = list()
    for i in range(0, t):
        l = sorted(projected[i], key=lambda x: x[0])
        l = list(map(lambda x: x[1], l))
        result.append(l)
    result = list(map(list, zip(*result)))
    return result


def pca(S, t, is_product):
    pca = PCA(n_components=t)
    X = np.array(S)
    pca.fit(X)
    if not is_product:
        print("Accumulative Variance Ratio : ", pca.explained_variance_ratio_.cumsum())
    return pd.DataFrame(pca.transform(X))


def SVD(S, t, is_product):
    svd = TruncatedSVD(n_components=t)
    X = np.array(S)
    svd.fit(X)
    if not is_product:
        print("Accumulative Variance Ratio : ", svd.explained_variance_ratio_.cumsum())
    return pd.DataFrame(svd.transform(X))