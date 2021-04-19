'''決定木学習モデル'''
'''ランダムフォレストモデル'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from logistic import plot_decision_regions, plot_decision_regions2
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier


# irisのデータセットをロード
iris = datasets.load_iris()
# 3、4列目を抽出
X = iris.data[:, [2, 3]]
y = iris.target


# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# ジニ不純度を指標とする決定気のインスタンスを生成
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

# 決定木のモデルを訓練データに適合させる
tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions2(X_combined, y_combined, classifier=tree_model,
                      test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 訓練後の決定木モデルの可視化
tree.plot_tree(tree_model)
dot_data = export_graphviz(tree_model, filled=True, rounded=True,
    class_names=['Setosa', 'Versicolor', 'Virginica'],
    feature_names=['petal length', 'petal width'],
    out_file=None)
graph = graph_from_dot_data(dot_data)
#graph.write_png('tree.png')
plt.show()


'''ランダムフォレストモデル'''
# ジニ不純度を指標とするランダムフォレストのインスタンスを生成
forest = RandomForestClassifier(criterion='gini',
    n_estimators=25, random_state=1, n_jobs=2)

# 訓練データにランダムフォレストモデルを適合させる
forest.fit(X_train, y_train)
plot_decision_regions2(X_combined, y_combined,
    classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
