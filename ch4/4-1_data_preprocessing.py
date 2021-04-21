import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer


# サンプルデータの作成
csv_data = '''
    A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''

# サンプルデータを読み込む
df = pd.read_csv(StringIO(csv_data))
print(df)

# 各特徴量の欠測値をカウント
df.isnull().sum()

'''欠測値を削除'''
# 欠測値を含む行を削除
df.dropna()
# 欠測値を含む列を削除
df.dropna(axis=1)
# 全ての列がNaNである行だけを削除
df.dropna(how='all')
# 非NaN値が4つ未満の行を削除
df.dropna(thresh=4)
# 特定の列(この場合は'C')のNaNが含まれている行だけを削除
df.dropna(subset=['C'])


'''欠測値を補完'''
# 平均値補完
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
# データを適合
imr = imr.fit(df.values)
# 補完を実行
imputed_data = imr.transform(df.values)
print('平均値補完\n', imputed_data)

# pandasでも可能
print('pandasでの平均値補完\n')
print(df.fillna(df.mean()))
