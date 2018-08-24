#!/usr/bin/env python3
import copy
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from examples.modules import utils, dimension_reduction as dim_red, evaluation as eval, clustering as cluster
import sys
from sklearn.metrics import roc_auc_score


try:
    path = sys.argv[1]
except IndexError:
    is_product = False
else:
    is_product = True

DIMENSION = 20
SAMPLE_RATE = 0.05
CONTAMINATION = [0.001, 0.002, 0.003, 0.004, 0.005, 0.008,  0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
N_NEIGHBOURS_SEARCH_RANGE = range(1, 20)
SEPARATOR = "==============================\n"

# 0. Data loading
if is_product:
    train, ytrain = utils.load_train_data(path, is_product)
else:
    train, ytrain = utils.load_train_data('../data_in/nmap_normal.csv', is_product)

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
    print("AUC score : ", roc_auc_score(ytrain, train["rate"]))
    print("finish")
