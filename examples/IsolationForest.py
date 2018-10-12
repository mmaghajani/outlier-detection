#!/usr/bin/env python3
import copy
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from modules import utils, dimension_reduction as dim_red, evaluation as eval, clustering as cluster
import sys
from sklearn.metrics import roc_auc_score, roc_curve

try:
    path = sys.argv[1]
except IndexError:
    is_product = False
else:
    is_product = True

DIMENSION = 20
SEPARATOR = "==============================\n"

# 0. Data loading
if is_product:
    train, ytrain = utils.load_train_data(path, is_product)
else:
    train, ytrain = utils.load_train_data('../data_in/u2r.csv', is_product)

# 1. Dimension Reduction
T = DIMENSION
n = train.shape[0]
projected = dim_red.pca(train, T, is_product)


# 2. Clustering
train["rate"] = cluster.isolation_forest_score(projected)
train["label"] = ytrain

# 3. Evaluation
if is_product:
    for i in train["rate"]:
        print(i)
else:
    fpr, tpr, threshold = roc_curve(ytrain, train["rate"])
    t = np.arange(0., 5., 0.001)
    utils.plot(1, 1, fpr, tpr, 'b', t, t, 'r')
    print("AUC score : ", roc_auc_score(ytrain, train["rate"]))
    print("finish")
