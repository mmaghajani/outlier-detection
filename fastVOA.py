import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules import dimension_reduction as dim_red
from modules import clustering as cluster
from modules import evaluation as eval


def sample(record_number, train):
    origin_train = train
    origin_train["label"] = ytrain
    outliers = origin_train[origin_train["label"] == 1]
    normal = origin_train[origin_train["label"] == 0]
    outliers = outliers.sample(frac=1)
    outliers = outliers[:10]
    normal = normal[:record_number - 10]
    data = pd.concat([outliers, normal])
    return pd.DataFrame(data)


def plot(axisX, axisY, list1, list2, color, list12=[], list22=[], color2=None):
    if list12 is not []:
        plt.plot(list1, list2, color + 'o', list12, list22, color2 + 's')
        plt.axis([0, axisX, 0, axisY])
        plt.show()
    else:
        plt.plot(list1, list2, color + 'o')
        plt.axis([0, axisX, 0, axisY])
        plt.show()


# 0. Data loading
train_url = "./data_in/nmap_normal.csv"
train = pd.read_csv(train_url, delimiter=',', header=None)
ytrain = train.iloc[:, -1]
train = train[:-1]
print("data is loaded")

T = 5
# 1. Dimension Reduction
n = train.shape[0]
projected = dim_red.random_projection(train, T)

# 2. Clustering
train["rate"] = cluster.fast_voa(projected, n, T, 5, 5)
# print(train["rate"])
train["label"] = ytrain

# 3. Evaluation
roc = eval.get_ROC(train)
t = np.arange(0., 5., 0.01)
plot(1, 1, roc[1], roc[0], 'b', t, t, 'r')

print("finish")
