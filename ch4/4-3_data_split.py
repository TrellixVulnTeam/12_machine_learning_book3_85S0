import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# クラスラベルを表示
print('Class labels', np.unique(df_wine['Class label']))
print('先頭の5行を表示\n', df_wine.head())

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


'''L1正則化'''
# L1ロジスティック回帰のインスタンス
# Cの値によって正則化の効果を強めたりできる
lr = LogisticRegression(
    penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')

# 訓練データに適合
lr.fit(X_train_std, y_train)

# 訓練データに対する正解率の表示
print('Training accuracy:', lr.score(X_train_std, y_train))
# テストデータに対する正解率の表示
print('Test accuracy:', lr.score(X_test_std, y_test))

# 重み(w0)の表示
print('重み係数w0\n'. lr.intercept_)

# 重み係数の表示
print('重み係数\n', lr.coef_)


'''正則化の強さを変化させる'''

# 描画の準備
fig = plt.figure()
ax = plt.subplot(111)

# 各係数の色リスト
colors = ['blue', 'green', 'red', 'cyan', 'magenta',
    'yellow', 'black', 'pink', 'lightgreen', 'lightblue',
    'gray', 'indigo', 'orange']
# 空のリストを生成 (重み係数、逆正則化パラメータ)
weights, params = [], []

# 逆正則化パラメータの値ごとに処理
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear',
    multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

# 重み係数をnumpy配列に変換
weights = np.array(weights)
# 各重みをプロット
for column, color in zip(range(weights.shape[1]), colors):
    # 横軸を逆正則化パラメータ、縦軸を重み係数とした折れ線グラフ
    plt.plot(params, weights[:, column],
        label=df_wine.columns[column + 1], color=color)

# y=0に黒い波線を引く
plt.axhline(0, color='black', linestyle='--', linewidth=3)
# 横軸の範囲を設定
plt.xlim([10**(-5), 10**5])
# 軸ラベルの設定
plt.ylabel('weight coefficient')
plt.xlabel('C')
# 横軸を対数スケールに設定
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1,
    fancybox=True)
plt.show()
