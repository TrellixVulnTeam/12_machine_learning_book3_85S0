from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
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
