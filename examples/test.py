import pandas as pd
train = pd.read_csv('../data_in/KDDTest2.csv', delimiter=',', header=None)
b = list()
for row in train.iterrows():
    b.append(list())

for name, column in train.iteritems():
    # max = max(column)
    a = list()
    print(index)
    for i in column:
        print(i)
