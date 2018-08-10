import numpy as np
import matplotlib.pyplot as plt
import itertools


def get_ROC(train):
    tp = fn = fp = tn = tpr = fpr = 0
    result = train["rate"]
    label = train["label"]
    tpr_list = []
    fpr_list = []

    for tr in range(int(np.min(result)), int(np.max(result)) + 1):
        for index, i in train.iterrows():
            if result[index] < tr:  # outlier
                if label[index] == 1:
                    tp += 1
                else:
                    fp += 1
            else:  # normal
                if label[index] == 1:
                    fn += 1
                else:
                    tn += 1
        # print(tp, fn, fp, tn)
        if tp == 0 and fn == 0:
            tpr = 0
        else:
            tpr = tp / (tp + fn)

        if fp == 0 and tn == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         print("Normalized confusion matrix")
    else:
        #         print('Confusion matrix, without normalization')
        tmp = 2

    #     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_precision_recall_fscore(confusion_matrix):
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = (2 * precision * recall) / (precision + recall)
    return precision, recall, fscore
