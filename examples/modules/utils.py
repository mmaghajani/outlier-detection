import pandas as pd
import matplotlib.pyplot as plt


def load_train_data(train_url, is_product):
    train = pd.read_csv(train_url, delimiter=',', header=None)
    ytrain = train.iloc[:, -1]
    train = train.drop(columns=[train.shape[1] - 1])
    if not is_product:
        print("Data Is Loaded\n", "==============================\n")
    return train, ytrain


def sample(data_proportion, train):
    data_size = int(len(train) * data_proportion)

    outliers = train[train["label"] == 1]
    normal = train[train["label"] == 0]
    outliers = outliers.sample(frac=1)
    normal = normal.sample(frac=1)
    if len(outliers) < 10:
        normal = normal[:data_size-len(outliers)]
    else:
        normal_data_size = int(len(normal)*data_proportion)
        outlier_data_size = int(len(outliers)*data_proportion)
        normal = normal[:normal_data_size]
        outliers = outliers[:outlier_data_size]

    data = pd.concat([outliers, normal])
    data = data.sample(frac=1)
    return pd.DataFrame(data)


def plot(axisX, axisY, list1, list2, color, list12=[], list22=[], color2=None):
    if list12 is not []:
        plt.plot(list1, list2, color + 'o', list12, list22, color2 + 's', markersize=1, linewidth=3)
        plt.axis([0, axisX, 0, axisY])
        plt.show()
    else:
        plt.plot(list1, list2, color + 'o', markersize=1, linewidth=3)
        plt.axis([0, axisX, 0, axisY])
        plt.show()