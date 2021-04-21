import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


#%%
'''カテゴリデータのエンコーディング'''
# サンプルデータを生成(Tシャツの色、サイズ、価格、クラスラベル)
df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']])
# 列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

#%%
'''順序特徴量のマッピング'''
# Tシャツのサイズと整数を対応させるディクショナリーを作成
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
# Tシャツのサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
print('変換後\n', df)

#%%
'''クラスラベルのエンコーディング'''
# クラスラベルと整数を対応させるディクショナリーを生成
class_mapping = {label: idx for idx, label in
    enumerate(np.unique(df['classlabel']))}
print('クラスラベルと整数の対応\n', class_mapping)
# クラスラベルを整数に変換
df['classlabel'] = df['classlabel'].map(class_mapping)
print('変換後\n', df)

# クラスを用いたラベルエンコーダ
# ラベルエンコーダのインスタンスを生成
class_le = LabelEncoder()
# クラスラベルから整数に変換
y = class_le.fit_transform(df['classlabel'].values)
print('クラスラベルを整数に変換', y)
# クラスラベルを文字列にもどす
print('文字列に直す', class_le.inverse_transform(y))


'''名義特徴量でのone-hot,エンコーディング'''
# Tシャツのいろ、サイズ、価格を抽出
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print('one-hotエンコード\n', X)

X = df[['color', 'size', 'price']].values
# one-hotエンコーダーの生成
color_ohe = OneHotEncoder()
# one-hotエンコーダーを実行
print('one-hotエンコード後')
print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())

# ダミー特徴量を生成
print('ダミー特徴量を生成')
print(pd.get_dummies(df[['price', 'color', 'size']]))

# 特徴量の列を一つ削除して作成する
# 1列削除しても判別できる
print('1列削除ver.\n')
print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))
