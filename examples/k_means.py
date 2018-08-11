from distutils.command.config import config
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import pandas as pd
import matplotlib.pyplot as plt
from modules import dimension_reduction as dim_red
from modules import clustering as cluster
from modules import evaluation as eval
from mlxtend.evaluate import confusion_matrix
from modules import utils
import copy

DIMENSION = 7
SAMPLE_RATE = 0.1
N_CLUSTERS_SEARCH_RANGE = range(8, 12)
OUTLIER_CLUSTER_SIZE_LIMIT_RANGE = range(40, 60)
SEPARATOR = "==============================\n"

# 0. Data loading
train_url = '../data_in/global.csv'
train = pd.read_csv(train_url, delimiter=',', header=None)
ytrain = train.iloc[:, -1]
train = train[:-1]
print("data is loaded")

# 1. Dimension Reduction
T = DIMENSION
n = train.shape[0]
projected = dim_red.pca(train, T)

# 2. parameter tuning
temp = copy.deepcopy(projected)
temp["label"] = ytrain
sample = utils.sample(SAMPLE_RATE, temp)
sample_label = sample.iloc[:, -1]
sample = sample.drop(columns=['label'])
fscore_max = 0
for outlier_cluster_size_limit in OUTLIER_CLUSTER_SIZE_LIMIT_RANGE:
    for n_clusters_samples in N_CLUSTERS_SEARCH_RANGE:
        temp_sample = copy.deepcopy(sample)
        predict = cluster.k_means(temp_sample, n_clusters_samples, outlier_cluster_size_limit)
        temp_sample["predict"] = predict
        temp_sample["label"] = sample_label
        classes = [0, 1]
        confusion_matrix_all = confusion_matrix(temp_sample["label"], temp_sample["predict"], binary=True)
        _, _, fscore = eval.compute_precision_recall_fscore(confusion_matrix_all)
        if fscore > fscore_max:
            fscore_max = fscore
            best_n_cluster_samples = n_clusters_samples
            best_outlier_cluster_size_limit = outlier_cluster_size_limit

print("Parameter Tuning Completed => best n samples : ", best_n_cluster_samples,
      " best outlier cluster size limit : ", best_outlier_cluster_size_limit, "\n", SEPARATOR)


# 3. Clustering
train["predict"] = cluster.k_means(projected, best_n_cluster_samples, best_outlier_cluster_size_limit)
train["label"] = ytrain

# 3. Evaluation
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
