import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules import dimension_reduction as dim_red
from modules import clustering as cluster
from modules import evaluation as eval
import pprint
from mlxtend.evaluate import confusion_matrix


# 0. Data loading
train_url = "./data_in/ipsweep_normal.csv"
train = pd.read_csv(train_url, delimiter=',', header=None)
ytrain = train.iloc[:, -1]
train = train[:-1]
print("data is loaded")

T = 10
# 1. Dimension Reduction
n = train.shape[0]
projected = dim_red.pca(train, T)

# 2. Clustering
# new_projected = dim_red.prepare_projected_data(projected, T)
predict = cluster.DB_Scan(projected, 40, 2)
print(predict)
index = 0
for i in predict:
    if i < 0:
        predict[index] = 1
    else:
        predict[index] = 0
    index += 1
print(predict)
train["predict"] = predict
train["label"] = ytrain

# 3. Evaluation
classes = [0, 1]
confusion_matrix_all = confusion_matrix(train["label"], train["predict"], binary=True)
plt.figure()
eval.plot_confusion_matrix(confusion_matrix_all, classes, normalize=False)
plt.show()

print("finish")
