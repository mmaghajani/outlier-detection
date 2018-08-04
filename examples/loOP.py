import numpy as np
import pandas as pd
from modules import dimension_reduction as dim_red
from modules import clustering as cluster
from modules import evaluation as eval
from modules import utils


# 0. Data loading
train_url = "./data_in/r2l.csv"
train = pd.read_csv(train_url, delimiter=',', header=None)
ytrain = train.iloc[:, -1]
train = train[:-1]
print("data is loaded")

T = 5
# 1. Dimension Reduction
n = train.shape[0]
projected = dim_red.SVD(train, T)

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
roc = eval.get_ROC(train)
t = np.arange(0., 5., 0.01)
utils.plot(1, 1, roc[1], roc[0], 'b', t, t, 'r')

print("finish")
