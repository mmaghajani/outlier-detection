#!/usr/bin/env python3 -W ignore::DeprecationWarning
import warnings

from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using a non-tuple sequence")
import numpy as np
import pandas as pd
from examples.modules import utils, dimension_reduction as dim_red, evaluation as eval, clustering as cluster
import sys

DIMENSION = 20

try:
    path = sys.argv[1]
except IndexError:
    is_product = False
else:
    is_product = True


# 0. Data loading
if is_product:
    train, ytrain = utils.load_train_data(path, is_product)
else:
    train, ytrain = utils.load_train_data('../data_in/r2l.csv', is_product)

# 1. Dimension Reduction
T = DIMENSION
n = train.shape[0]
projected = dim_red.random_projection(train, T)

# 2. Clustering
train["rate"] = cluster.fast_voa(projected, n, T, 1, 1)
train["label"] = ytrain

# 3. Evaluation
if is_product:
    scores = train["rate"]
    scores = list(map(lambda x: float(x), scores))
    max = max(scores)
    for i in range(len(scores)):
        scores[i] = scores[i] / max
    for i in scores:
        x = i*1000
        x = int(x)
        print(x/1000)
else:
    scores = train["rate"]
    scores = list(map(lambda x: float(x), scores))
    max = max(scores)
    for i in range(len(scores)):
        scores[i] = scores[i] / max
        # scores[i] = scores[i] * 1000
        # scores[i] = int(scores[i])
        # scores[i] = scores[i] / 1000

    print("AUC score : ", roc_auc_score(ytrain, scores))
    print("finish")
