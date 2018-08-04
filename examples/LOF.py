import pandas as pd
import matplotlib.pyplot as plt
from modules import dimension_reduction as dim_red
from modules import clustering as cluster
from modules import evaluation as eval
from mlxtend.evaluate import confusion_matrix


# 0. Data loading
train_url = "./data_in/r2l.csv"
train = pd.read_csv(train_url, delimiter=',', header=None)
ytrain = train.iloc[:, -1]
# print(train)
print("data is loaded")

T = 7
# 1. Dimension Reduction
n = train.shape[0]
projected = dim_red.SVD(train, T)
# print(projected)

# 2. Clustering
# new_projected = dim_red.prepare_projected_data(projected, T)
predict = cluster.LOF(projected, 20)
for i in range(len(predict)):
    if predict[i] == -1:
        predict[i] = 1
    else:
        predict[i] = 0
train["predict"] = predict
train["label"] = ytrain

# 3. Evaluation
classes = [0, 1]
confusion_matrix_all = confusion_matrix(train["label"], train["predict"], binary=True)
plt.figure()
eval.plot_confusion_matrix(confusion_matrix_all, classes, normalize=False)
plt.show()

print("finish")
