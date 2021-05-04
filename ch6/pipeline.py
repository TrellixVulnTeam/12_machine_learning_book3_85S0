'''パイプライン'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


# データの準備
df = pd.read_csv('https://archive.ics.uci.edu/ml/\
machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
header = None)

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
