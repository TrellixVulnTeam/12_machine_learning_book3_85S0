'''パイプライン'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from distutils.version import LooseVersion as Version
from scipy import interp
from sklearn.pipeline import make_pipeline


# データの準備
df = pd.read_csv('/Users/rukaoide/Library/Mobile Documents/\
com~apple~CloudDocs/Documents/Python/12_machine_learning_book3/\
ch6/wdbc.txt', header = None)

df.head()
df.shape

# 特徴量の割り当て
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_

le.transform(['M', 'B'])

# 訓練データとテストデータへ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=1)

# パイプラインで変換器を推定器を結合する
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2),
    LogisticRegression(random_state=1, solver='lbfgs'))
#%%
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))


#%% 層k分割交差法
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' %
        (k+1,np.bincount(y_train[train]), score))

print('\nCV accuracy: %.3f +/- %.3f' %
    (np.mean(scores), np.std(scores)))


#%% k分割交差検証の性能指数を算出
# cv=分割数、n_jobs=CPU数
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train,
    cv=10, n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


#%% 学習曲線と検証曲線
# 学習曲線
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(
    penalty='l2', random_state=1, solver='lbfgs', max_iter=10000))

train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_lr, X=X_train, y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o',
    markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean+train_std,
    train_mean-train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,color='green', linestyle='--',
    marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(train_sizes, test_mean+test_std,
    test_mean-test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
plt.show()

#%% 検証曲線
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(
    estimator=pipe_lr, X=X_train, y=y_train,
    param_name='logisticregression__C', param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o',
    markersize=5, label='Training accuracy')
plt.fill_between(param_range, train_mean+train_std,
    train_mean-train_std, alpha=0.15, color='blue')

plt.plot(param_range, test_mean, color='green', linestyle='--',
    marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(param_range, test_mean + test_std,
    test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
plt.show()


#%% グリッドサーチによるハイパーパラメータのチューニング
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# ハイパーパラメータのリスト
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid,
    scoring='accuracy', refit=True, cv=10, n_jobs=-1)
gs = gs.fit(X_train, y_train)

# モデルの最良スコアを出力
print(gs.best_score_)
# 最良スコアとなるパラメータ値を出力
print(gs.best_params_)

# テストデータセットによるモデルの評価
clf = gs.best_estimator_
# gsのrefit=Trueならいらない
# clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))


#%% 入れ子式の交差検証
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid,
    scoring='accuracy', cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy',
    cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# 決定木分類器との比較
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
    param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
    scoring='accuracy', cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


#%% 混合行列
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
# テストと予測データから混合行列を生成
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

# グラブ化する
# 図のサイズを指定する
fig, ax = plt.subplots(figsize=(2.5, 2.5))
# matshow関数で行列からヒートマップを描画
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.show()

# 適合率、再現率、F1スコアを出力
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))


#%% ROC曲線をプロット
# スケーリング、主成分分析、ロジスティック回帰を指定
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2),
    LogisticRegression(penalty='l2', random_state=1, solver='lbfgs', C=100.0))

# 2つの特徴量を抽出
X_train2 = X_train[:, [4, 14]]

# 層化k分割交差検証イテレータを表すクラスをインスタンス化
cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0

# 0から1までの間で100個の要素を生成
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    # predict_probaで確率を予測。fitで適合させる。
    probas = pipe_lr.fit(
        X_train2[train], y_train[train]).predict_proba(X_train2[test])
    # roc_curveでROC曲線の性能を計算してプロット
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    # FPR(x軸)とTPR(y軸)を線形補間
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    # 曲線下面積(AUC)を計算
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,
        label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

# 当て推量をプロット
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6,0.6,0.6),
    label='Random guessing')
# FPR、TPR、ROC、AUSそれぞれの平均を計算してプロット
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
    label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
# 完全に予測が正解した時のROC曲線をプロット
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black',
    label='Perfect performance')
# グラブの各項目を設定
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


#%% クラスの不均衡に対処する
# 不均衡なデータセットの作成
X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

# クラスの不均衡具合を表示
y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100

# 少数派クラスのアップサンプリング
# アップサンプリングする前のクラス1のデータ個数
print('Number of class 1 examples before:', X_imb[y_imb == 1].shape[0])
# データ点の個数がクラス0と同じになるまで新しいデータ点を復元抽出
X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
    y_imb[y_imb == 1], replace=True, n_samples=X_imb[y_imb == 0].shape[0],
    random_state=123)
# アップサンプリングした後のクラス1のデータ個数
print('Number of class 1 examples after:', X_upsampled.shape[0])

# 均衡なデータセットの生成
X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))

# クラスの均衡具合を表示
y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100
