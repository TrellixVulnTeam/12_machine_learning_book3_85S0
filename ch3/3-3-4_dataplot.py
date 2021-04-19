'''ロジスティック回帰モデル(sk-learn)'''

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from logistic import plot_decision_regions2


# irisのデータセットをロード
iris = datasets.load_iris()
# 3、4列目を抽出
X = iris.data[:, [2, 3]]
y = iris.target

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# 訓練データの平均と標準偏差を計算
sc = StandardScaler()
sc.fit(X_train)
# 平均と標準偏差を用いて標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# 訓練データとテストデータの結合
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

# 決定領域をプロット
plot_decision_regions2(X_combined_std, y_combined, classifier=lr,
    test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# テストデータセットのデータ点の所属確率
pre = lr.predict_proba(X_test_std[:3, :])
print('所属確率:', pre)
print('確率の和:', pre.sum(axis=1))
# クラスラベルの予測値の取得
print('所属ラベル:', pre.argmax(axis=1))
print('所属ラベル:', lr.predict(X_test_std[:3, :]))
# 単一データ点のクラスラベル予測
print('所属ラベル(単一データ点):', lr.predict(X_test_std[0, :].reshape(1, -1)))
