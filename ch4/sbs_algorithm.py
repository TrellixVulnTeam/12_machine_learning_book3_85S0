from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS:
    '''逐次後退選択を実行するクラス'''

    def __init__(self, estimator, k_features, scoring=accuracy_score,
        test_size=0.25, random_state=1):
        self.scoring = scoring             # 特徴量を評価する指標
        self.estimator = clone(estimator)  # 推定器
        self.k_features = k_features       # 洗濯する特徴量の個数
        self.test_size = test_size         # テストデータの割合
        self.random_state = random_state   # 乱数シードを固定する

    def fit(self, X, y):
        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        # 全ての特徴量の個数、列インデックる
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        # 全ての特徴量を用いてづコアを算出
        score = self._calc_score(X_train, y_train,
            X_test, y_test, self.indices_)
        # スコアを格納
        self.scores_ = [score]
