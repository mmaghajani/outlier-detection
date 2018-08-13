import copy
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import matplotlib.pyplot as plt
from modules import dimension_reduction as dim_red
from modules import clustering as cluster
from modules import evaluation as eval
from mlxtend.evaluate import confusion_matrix
from modules import utils

DIMENSION = 7
SAMPLE_RATE = 0.1
CONTAMINATION = [0.001, 0.002, 0.003, 0.004, 0.005, 0.008,  0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
N_NEIGHBOURS_SEARCH_RANGE = range(1, 20)
SEPARATOR = "==============================\n"

# 0. Data loading
train, ytrain = utils.load_train_data('../data_in/global.csv')

# 1. Dimension Reduction
T = DIMENSION
n = train.shape[0]
projected = dim_red.SVD(train, T)

# 2. parameter tuning
temp = copy.deepcopy(projected)
temp["label"] = ytrain
sample = utils.sample(SAMPLE_RATE, temp)
sample_label = sample.iloc[:, -1]
sample = sample.drop(columns=['label'])
fscore_max = 0
for contamination in CONTAMINATION:
    temp_sample = copy.deepcopy(sample)
    predict = cluster.isolation_forest(temp_sample, contamination)
    temp_sample["predict"] = predict
    temp_sample["label"] = sample_label
    classes = [0, 1]
    confusion_matrix_all = confusion_matrix(temp_sample["label"], temp_sample["predict"], binary=True)
    _, _, fscore = eval.compute_precision_recall_fscore(confusion_matrix_all)
    if fscore > fscore_max:
        fscore_max = fscore
        best_contamination = contamination

print("Parameter Tuning Completed => best contamination : ", best_contamination, "\n", SEPARATOR)

# 2. Clustering
train["predict"] = cluster.isolation_forest(projected, best_contamination)
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
