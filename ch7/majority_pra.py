from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from majority_vote_classifier import MajorityVoteClassifier


#%% irisデータセットの準備
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
y
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1, stratify=y)


#%% 各分類器の訓練データセットでの性能を評価
# モデル準備
clf1 = LogisticRegression(
    penalty='l2', C=0.001, solver='lbfgs', random_state=1)

clf2 = DecisionTreeClassifier(
    max_depth=1, criterion='entropy', random_state=0)

clf3 = KNeighborsClassifier(
    n_neighbors=1, p=2, metric='minkowski')

# 前処理クラス + 分類器
pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

clf_labels = ['Logistic regression', 'Decision tree', 'KNN']
print('10-fold cross validation:\n')

# モデル評価
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train,
        cv=10, scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
        % (scores.mean(), scores.std(), label))


#%% 多数決を用いてクラスラベルの予測
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train,
        cv=10, scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
        % (scores.mean(), scores.std(), label))


# テストデータセットからROC曲線を計算
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']

for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # 陽性クラスのラベルは1であることが前提
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls,
        label='%s(auc=%0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.show()


# アンサンブル分類器の決定領域の可視化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

# 決定領域を描画する最小値、最大値を生成
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
# グリッドポイントを生成
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1))
# 描画領域を2行2列に分割
f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row',
    figsize=(7, 5))
# 決定領域のプロット、青や赤の散布図の作成などを実行
# 変数idxは各分類器を描画する行と列の位置を表すタプル
for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
        X_train_std[y_train==0, 1], c='blue', marker='^', s=50)

    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],
        X_train_std[y_train==1, 1], c='green', marker='o', s=50)
    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -5., s='Sepal width [standardized]',
    ha='center', va='center', fontsize=12)
plt.text(-12.5, 4.5, s='Petal length [standardized]',
    ha='center', va='center', fontsize=12, rotation=90)
plt.show()


# ロジスティック回帰分類器のCと決定木の深さをチューニング
params = {'decisiontreeclassifier__max_depth': [1, 2],
    'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10,
    iid=False, scoring='roc_auc')
grid.fit(X_train, y_train)

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print('%0.3f +/- %0.2f %r' %
        (grid.cv_results_['mean_test_score'][r],
        grid.cv_results_['std_test_score'][r] / 2.0,
        grid.cv_results_['params'][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)
