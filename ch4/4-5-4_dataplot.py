from sbs_algorithm import SBS
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


'''訓練データとテストデータに分ける'''
# ワインのデータセットを読み込む
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
    'ml/machine-learning-databases/wine/wine.data', header=None)
# 列名を設定
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium', 'Total phenols',
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# 特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 訓練データとテストデータに分割(30%をテストデータ)
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3, random_state=0, stratify=y)

'''特徴量の尺度を揃える'''
#min-maxスケーリングのインスタンスを生成
mms = MinMaxScaler()
# 訓練データをスケーリング
X_train_norm = mms.fit_transform(X_train)
# テストデータをスケーリング
X_test_norm = mms.transform(X_test)

# 標準化のインスタンスを生成(平均=0, 標準偏差=1に変換)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# k最近傍法分類器のインスタンスを生成(近傍点数 = 5)
knn = KNeighborsClassifier(n_neighbors=5)
# 逐次後退選択のインスタンスを生成(特徴量の個数が1になるまで特徴量を選択)
sbs = SBS(knn, k_features=1)
# 逐次後退選択を実行
sbs.fit(X_train_std, y_train)
# 特徴量の個数リスト(13, 12, 11, ..., 1)
k_feat = [len(k) for k in sbs.subsets_]

# 横軸を特徴量の個数、縦軸をスコアとした折れ線グラフのプロット
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# 最小限の特徴量部分集合(k=3)の確認
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])


# 13個全ての特徴量を用いてモデルを適合
knn.fit(X_train_std, y_train)
# 訓練の正解率を出力
print('Training accuracy:', knn.score(X_train_std, y_train))
# テストの正解率を出力
print('Test accuracy:', knn.score(X_test_std, y_test))

# 3つの特徴量を用いてモデルを適合
knn.fit(X_train_std[:, k3], y_train)
# 訓練の正解率を出力
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
# テストの正解率を出力
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))
