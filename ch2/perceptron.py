import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    '''パーセプトロンの分類器'''

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        # 学習率
        self.eta = eta
        # 訓練データの訓練回数
        self.n_iter = n_iter
        # 重み初期化の乱数シード
        self.random_state = random_state

    def fit(self, X, y):
        '''訓練データに適応させる'''
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []

        # 訓練回数分まで訓練データを反復
        for _ in range(self.n_iter):
            errors = 0
            # 訓練データで重みを更新
            for xi, target in zip(X, y):
                # 重みの更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 重み[0]の更新
                self.w_[0] += update * 1
                # 重みの更新が0でない場合は誤分類としてカウント
                errors += int(update != 0.0)
            # 反復ごとの誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        '''総入力を計算'''
        return X @ self.w_[1:] + self.w_[0]

    def predict(self, X):
        '''1ステップ後のクラスラベルを返す'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# 決定境界の可視化
def plot_dicision_regions(X, y, classifier, resolution=0.02):
    # マーカーとカラーアップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定境界のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
    # 特徴量を1次元配列にして予測
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとに訓練データをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
            alpha=0.8, c=colors[idx], marker = markers[idx],
            label=cl, edgecolors='black')
