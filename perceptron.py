import numpy as np

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
