'''パーセプロトンアルゴリズム'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from perceptron import Perceptron
from perceptron import plot_dicision_regions


# データの読み込み
s = 'https://archive.ics.uci.edu/ml/\
machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

# 1〜100行目の目的変数の抽出と変換
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# 1〜100行目の1、3列目(今回使うデータ)を抽出
X = df.iloc[0:100, [0, 2]].values

# データのプロット
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# 学習とその様子
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

plot_dicision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
