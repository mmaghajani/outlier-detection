import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

from examples.modules import utils, dimension_reduction as dim_red, evaluation as eval, clustering as cluster

# 0. Data loading
train_url = "../data_in/r2l.csv"
train = pd.read_csv(train_url, delimiter=',', header=None)
ytrain = train.iloc[:, -1]
train = train[:-1]
print("data is loaded")

try:
    path = sys.argv[1]
except IndexError:
    is_product = False
else:
    is_product = True

T = 5
# 1. Dimension Reduction
n = train.shape[0]
projected = dim_red.SVD(train, T, is_product)

# 2. Clustering
# projected = dim_red.prepare_projected_data(projected, T)
predict = cluster.loOP(projected, 20)
# for i in range(len(predict)):
#     if predict[i] == -1:
#         predict[i] = 1
#     else:
#         predict[i] = 0
train["rate"] = predict
train["label"] = ytrain

# 3. Evaluation
fpr, tpr, threshold = roc_curve(ytrain, train["rate"])
t = np.arange(0., 5., 0.001)
utils.plot(1, 1, fpr, tpr, 'b', t, t, 'r')
print("AUC score : ", roc_auc_score(ytrain, train["rate"]))
print("finish")
