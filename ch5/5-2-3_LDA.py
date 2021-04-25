import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skPCA import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


'''線形判別分析（LDA）'''
#%%
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/wine/wine.data',header=None)

# 列項目のラベル付け
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium', 'Total phenols',
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue','OD280/OD315 of diluted wines', 'Proline']

# 最初の5行を表示
df_wine.head()

# 2列目以降のデータをXに、1列目のデータをyに格納
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# データを分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0)

# 平均と標準偏差を用いて標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# 平均ベクトルの算出
np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('MV %s: %s\n' % (label, mean_vecs[label - 1]))

#%% クラス内変動行列の計算
d = 13     # 特徴量の個数
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Within-class scatter matrix: %sx%s'
    % (S_W.shape[0], S_W.shape[1]))

#%% クラス間変動行列
mean_overall = np.mean(X_train_std, axis=0)
d = 13     # 特徴量の個数
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i+1, :].shape[0]
    # 列ベクトルの作成
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall) @ (mean_vec - mean_overall).T
print('Between-class scatter matrix: %sx%s'
    % (S_B.shape[0], S_B.shape[1]))

#%% 新しい特徴量の選択
# inv関数で逆行列、eig関数で固有値を計算
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)

# 固有対のタプルリストの作成
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
    for i in range(len(eigen_vals))]
# 固有対のリストを降順でソート
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

#%% 固有値を減らしながら分散説明率をプロットする
# 固有値の実数部分の総和を求める
tot = sum(eigen_vals.real)
# 分散説明率とその累乗話を計算
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align='center',
    label='Individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',
    label='Cumulative "discriminability"')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 最も効果のある2つの固有ベクトルから変換行列を作成
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
    eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

'''新しい特徴量空間にデータ点を射影する'''
#%% 標準化した訓練データに変換行列をかける
X_train_lda = X_train_std @ w
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
        X_train_lda[y_train == l, 1] * (-1),
        c=c, label=l, marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


'''sklearnによるLDA'''
#%% 次元数を指定してLDAのインスタンスを生成
lda = LDA(n_components=2)
# 訓練データの分類
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# テストデータの分類
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
