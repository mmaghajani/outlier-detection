import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import copy
import pandas as pd
import matplotlib.pyplot as plt
from modules import dimension_reduction as dim_red
from modules import clustering as cluster
from modules import evaluation as eval
from mlxtend.evaluate import confusion_matrix
from modules import utils


DIMENSION = 7
SAMPLE_RATE = 0.05
EPS_SEARCH_RANGE = (0.5, 15)
EPS_STEP = 0.5
MIN_SAMPLE_SEARCH_RANGE = range(1, 20)
SEPARATOR = "==============================\n"

# 0. Data loading
train, ytrain = utils.load_train_data('../data_in/r2l.csv')

# 1. Dimension Reduction
T = DIMENSION
n = train.shape[0]
projected = dim_red.SVD(train, T)
print("Dimension Reduction Completed\n", SEPARATOR)

# 2. parameter tuning
temp = copy.deepcopy(projected)
temp["label"] = ytrain
sample = utils.sample(SAMPLE_RATE, temp)
sample_label = sample.iloc[:, -1]
sample = sample.drop(columns=['label'])
fscore_max = 0
eps = EPS_SEARCH_RANGE[0]
while eps < EPS_SEARCH_RANGE[1]:
    for min_samples in MIN_SAMPLE_SEARCH_RANGE:
        temp_sample = copy.deepcopy(sample)
        predict = cluster.DB_Scan(temp_sample, eps, min_samples)
        temp_sample["predict"] = predict
        temp_sample["label"] = sample_label
        classes = [0, 1]
        confusion_matrix_all = confusion_matrix(temp_sample["label"], temp_sample["predict"], binary=True)
        _, _, fscore = eval.compute_precision_recall_fscore(confusion_matrix_all)
        if fscore > fscore_max:
            fscore_max = fscore
            best_eps = eps
            best_min_samples = min_samples
    eps += EPS_STEP
print("Parameter Tuning Completed => best eps : ", best_eps,
      " best min samples : ", best_min_samples, "\n", SEPARATOR)

# 3. Clustering
predict = cluster.DB_Scan(projected, best_eps, best_min_samples)
train["predict"] = predict
train["label"] = ytrain

# 4. Evaluation
classes = [0, 1]
confusion_matrix_all = confusion_matrix(train["label"], train["predict"], binary=True)
precision, recall, fscore = eval.compute_precision_recall_fscore(confusion_matrix_all)
print("Precision : ", precision)
print("Recall    : ", recall)
print("FScore    : ", fscore)
plt.figure()
eval.plot_confusion_matrix(confusion_matrix_all, classes, normalize=False)
plt.show()

print("finish")
