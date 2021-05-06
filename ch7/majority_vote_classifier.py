'''多数決アンサンブル分類器'''
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
import operator


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    # 多数決アンサンブル分類器
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value
            for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        # 分類器を学習させる
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                "; got (vote=%r)" % self.vote)
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                '; got %d weights, %d classifiers'
                % (len(self.weights), len(self.classifiers)))

        # LabelEncoderを使ってクラスラベルが0から始まるようにエンコードする
        # self.predictのnp.argmax呼び出しで重要になる
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            # 'Classlabel'での多数決
            # clf.predict呼び出しの結果を収集
            predictions = np.asarray([clf.predict(X)
                for clf in self.classifiers_]).T

            # 各データ点のクラス確率に重みをかけて足し合わせた値が最大となる列番号を返す
            maj_vote = np.apply_along_axis(lambda x:
                np.argmax(np.bincount(x, weights=self.weights)),
                axis=1, arr=predictions)
        # 各データ点に確率の最大値を与えるクラスラベルを抽出
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        # Xのクラス確率を予測する
        probas = np.asarray([clf.predict_proba(X)
            for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        # GridSearchの実行時に分類器のパラメータ名を取得
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            # キーを'分類器の名前__パラメータ名'、値をパラメータ名とする辞書の作成
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
            return out
