import numpy as np

class AdalineGD(object):
    '''ADAptive LInear NEuronの分類器'''

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
        self.cost_ = []

        # 訓練回数分まで訓練データを反復
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            # 活性化関数を作用
            output = self.activation(net_input)
            errors = (y - output)
            # 重みの更新
            self.w_[1:] += self.eta * X.T @ errors
            self.w_[0] += self.eta * 1 * errors.sum()
            # コスト関数
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        '''総入力を計算'''
        return X @ self.w_[1:] + self.w_[0]

    def activation(self, X):
        '''線形活性化関数の出力'''
        return X

    def predict(self, X):
        '''1ステップ後のクラスラベルを返す'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
