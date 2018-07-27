# TODO we do not check if center and a and b are in a line
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_url = "./data_in/global.csv"
train = pd.read_csv(train_url, delimiter=',', header=None)
train = train.sample(frac=1)
ytrain = train.iloc[:, -1]
print("data is loaded")

MOD_RATE = 15


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


def getABOF(vertex, a, b):
    va = a - vertex
    vb = b - vertex
    cosine_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
    angle = np.arccos(cosine_angle)
    angle_degree = np.degrees(angle)
    dista = np.linalg.norm(va)
    distb = np.linalg.norm(vb)
    return angle_degree


def plot(axisX, axisY, list1, list2, color, list12=[], list22=[], color2=None):
    if list12 is not []:
        plt.plot(list1, list2, color + 'o', list12, list22, color2 + 's')
        plt.axis([0, axisX, 0, axisY])
        plt.show()
    else:
        plt.plot(list1, list2, color + 'o')
        plt.axis([0, axisX, 0, axisY])
        plt.show()


def get_ROC(train):
    tp = fn = fp = tn = tpr = fpr = 0
    result = train["ABOF"]
    label = train["label"]
    print(result, label)
    tpr_list = []
    fpr_list = []

    for tr in range(int(np.min(result)), int(np.max(result))):
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


# train = hash_train(train_temp, 20)
# train.to_csv("hash_train.csv", index=False)
train = sample(100, train)
print(train)
varABOF = []
varAvg = []
varModABOF = []
varModAVG = []

for t, center in train.iterrows():
    if t % 10 == 0:
        print(t)
    centerABOF = []
    center = list(center)
    for index, i in train.iterrows():
        if center != list(i):
            for j in range(index, train.shape[0]):
                rowJ = list(train.iloc[j])
                if center != rowJ and list(i) != rowJ:
                    centerABOF.append(getABOF(center, np.array(list(i)), np.array(rowJ)))
    varABOF.append(np.var(centerABOF))
    varAvg.append(np.average(centerABOF))

for record in varABOF:
    varModABOF.append(np.remainder(record, MOD_RATE))

for record in varAvg:
    varModAVG.append(np.remainder(record, MOD_RATE))

train["ABOF"] = varABOF
train["avg"] = varAvg
train["mod_avg"] = varModAVG
train["mod_ABOF"] = varModABOF
train["label"] = ytrain
roc = get_ROC(train)
print(roc[0], roc[1])
t = np.arange(0., 5., 0.01)

plot(1, 1, roc[1], roc[0], 'b', t, t, 'r')

print("finish")
train.to_csv("./data_out/mammadAgha.csv", index=False)
