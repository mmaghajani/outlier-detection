import pandas as pd
import matplotlib.pyplot as plt


def sample(record_number, train):
    ytrain = train.iloc[:, -1]
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