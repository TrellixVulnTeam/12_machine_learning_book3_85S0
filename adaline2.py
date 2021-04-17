'''確率的勾配降下法'''

import numpy as np

class AdalineSGD(object):
    '''ADAptive LInear NEuronの分類器'''

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        # 学習率
        self.eta = eta
        # 訓練データの訓練回数
        self.n_iter = n_iter
        # 重みの初期化フラグ
        self.w_initialized = False
        # シャッフルするかどうかのフラグの初期化
        self.shuffle = shuffle
        # 重み初期化の乱数シード
        self.random_state = random_state

    def fit(self, X, y):
        '''訓練データに適応させる'''
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        # 訓練回数分まで訓練データを反復
        for _ in range(self.n_iter):
            # 重みベクトルの生成
            self._initialize_weights(X.shape[1])
            self.cost_ = []
            # 訓練回数分まで訓練データを反復
            for i in range(self.n_iter):
                if self.shuffle:
                    X, y = self._shuffle(X, y)
                # 各訓練データのコストを格納
                cost = []
                # 各訓練データに対する計算
                for xi, target in zip(X, y):
                    cost.append(self._update_weights(xi, target))
                avg_cost = sum(cost) / len(y)
                self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        '''重みを再初期化することなく訓練データに適応させる'''
        # 初期化されていない時は初期化を実行
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        # 目的変数yの要素数が2以上の時、各データで重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        # 目的変数yの要素数が1の時、訓練データ全体で重みを更新
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        '''訓練データをシャッフル'''
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        '''重みを小さな乱数に初期化'''
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        '''ADALINEの学習規則を用いて重みを更新'''
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * error * xi
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        '''総入力を計算'''
        return X @ self.w_[1:] + self.w_[0]

    def activation(self, X):
        '''線形活性化関数の出力'''
        return X

    def predict(self, X):
        '''1ステップ後のクラスラベルを返す'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
