import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix


#%% データの準備
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-3rd-edition/'
                 'master/ch10/housing.data.txt', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
    'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

#%% 散布図行列を作成
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
# 変数ペアの関係をプロット
scatterplotmatrix(df[cols].values, figsize=(10, 8), names=cols, alpha=0.5)
plt.tight_layout()
plt.show()
