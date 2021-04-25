import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skPCA import plot_decision_regions
from sklearn.linear_model import LogisticRegression


'''主成分分析(PCA)'''
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

# 共分散行列を作成
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)


'''分散説明率の可視化'''
#%% 固有値の合計
tot = sum(eigen_vals)
# 分散説明率を計算
var_exp = [i / tot for i in sorted(eigen_vals, reverse=True)]
# 分散説明率の累乗和を取得
cum_var_exp = np.cumsum(var_exp)

# 説明分散率の棒グラフを作成
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
    label='Individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
    label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


'''特徴量変換'''
#%% (固有値、固有ベクトル)のタプルリストを作成
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
    for i in range(len(eigen_vals))]
# (固有値、固有ベクトル)のタプルを大きいものから順に並び替え
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
    eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

# 変換後のデータセットを2次元の散布図でプロット
X_train_pca = X_train_std @ w
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

# クラスラベル、点の色、点の種類の組み合わせからなるリストを生成してプロット
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],
        X_train_pca[y_train == l, 1],
        c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


'''sklearnでのPCA'''
# 主成分数を指定してPCAインスタンスを生成
pca = PCA(n_components=2)
# 次元削減
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
# 削減したデータセットでロジスティック回帰モデルを適合
lr = lr.fit(X_train_pca, y_train)
# 決定境界をプロット
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# 決定境界をプロット
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

'''sklearnによる分散説明率の計算'''
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
var_data = pca.explained_variance_ratio_
plt.bar(range(1, 14), var_data, alpha=0.5, align='center',
    label='Individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
