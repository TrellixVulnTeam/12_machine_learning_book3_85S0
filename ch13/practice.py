import tensorflow as tf
import numpy as np
%precision 3


#%% テンソルの作成方法
a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]
t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

print(t_a)
print(t_b)

#%% テンソルかどうかの確認
tf.is_tensor(a), tf.is_tensor(t_a)

# テンソルの次元確認
t_ones = tf.ones((2, 3))
t_ones.shape

#%% テンソルの値にアクセス
t_ones.numpy()

#%% 定数値のテンソルを作成
const_tensor = tf.constant([1.2, 5, np.pi], dtype=tf.float32)
print(const_tensor)

#%% テンソルのデータ型の変更
t_a_new = tf.cast(t_a, tf.int64)
print(t_a_new.dtype)

# テンソルの転置
t = tf.random.uniform(shape=(3, 5))
t_tr = tf.transpose(t)
print(t.shape, '-->', t_tr.shape)

# テンソルの形状変化
t = tf.zeros((30, ))
t_reshape = tf.reshape(t, shape=(5, 6))
print(t_reshape.shape)

# 不要な次元の削除
t = tf.zeros((1, 2, 1, 4, 1))
t_sqz = tf.squeeze(t, axis=(2, 4))
print(t.shape, '-->', t_sqz.shape)


# テンソルでの算術演算
#%% サンプルの準備
tf.random.set_seed(1)
t1 = tf.random.uniform(shape=(5, 2), minval=-1.0, maxval=1.0)
t2 = tf.random.normal(shape=(5, 2), mean=0.0, stddev=1.0)
t3 = tf.multiply(t1, t2).numpy()
print(t3)

# 特定の軸に沿って平均などを求める
t4 = tf.math.reduce_mean(t1, axis=0)
print(t4)

# 行列の積
t5 = tf.linalg.matmul(t1, t2, transpose_b=True)
print(t5.numpy())

t6 = tf.linalg.matmul(t1, t2, transpose_a=True)
print(t6.numpy())

# ノルムの計算
norm_t1 = tf.norm(t1, ord=2, axis=1).numpy()
print(norm_t1)


#%% テンソルの分割、積み上げ、連結
# テンソルの分割
tf.random.set_seed(1)
t = tf.random.uniform((6, ))
print(t.numpy())

# tを3つに分割
t_splits = tf.split(t, 3)
[item.numpy() for item in t_splits]

# 異なるのサイズで分割
t = tf.random.uniform((5, ))
print(t.numpy())

t_splits = tf.split(t, num_or_size_splits=[3, 2])
[item.numpy() for item in t_splits]

# テンソルの積み上げ
A = tf.ones((3, ))
B = tf.zeros((3, ))

S = tf.stack([A, B], axis=0)
print(S.numpy())

# テンソルの連結
A = tf.ones((3, ))
B = tf.zeros((2, ))

C = tf.concat([A, B], axis=0)
print(C.numpy())

#
A = tf.ones((3, 3))
B = tf.zeros((3, 3))

C = tf.concat([A, B], axis=1)
print(C.numpy())


# Tensorflow Data API
#%% 既存のテンソルからDatasetを作成
a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]
ds = tf.data.Dataset.from_tensor_slices(a)
print(ds)

# Datasetから繰り返し処理を行う
for item in ds:
    print(item)

# バッチ数を指定して繰り返し処理
ds_batch = ds.batch(3)
for i, elem in enumerate(ds_batch, 1):
    print('batch{}:'.format(i), elem.numpy())


#%% 2つのテンソルを1つのデータセットにする
# サンプルの準備
tf.random.set_seed(1)

t_x = tf.random.uniform([4, 3], dtype=tf.float32)
t_y = tf.range(4)

# データセットの結合(その1)
ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)

ds_joint = tf.data.Dataset.zip((ds_x, ds_y))

for example in ds_joint:
    print('x:', example[0].numpy(),
            'y:', example[1].numpy())

# データセットの結合(その2)
ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y))

for example in ds_joint:
    print('x:', example[0].numpy(),
            'y:', example[1].numpy())

#%% 各データセットに対する処理
ds_trans = ds_joint.map(lambda x, y: (x*2 - 1.0, y))

for example in ds_trans:
    print('x:', example[0].numpy(),
            'y:', example[1].numpy())


#%% シャッフル、バッチ、リピート
tf.random.set_seed(1)
ds = ds_joint.shuffle(buffer_size=len(t_x))
for example in ds:
    print('x: ', example[0].numpy(),
            'y: ', example[1].numpy())

# バッチで取り出す
ds = ds_joint.batch(batch_size=3, drop_remainder=False)
batch_x, batch_y = next(iter(ds))
print('Batch-x:\n', batch_x.numpy())
print('Batch-y:', batch_y.numpy())

# リピートで繰り返す
ds = ds_joint.batch(3).repeat(count=2)
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# リピート → バッチだと結果が変わる
ds = ds_joint.repeat(count=2).batch(3)
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# バッチ、リピート、シャッフルを色々な順番で
# シャッフル → バッチ → リピート
ds = ds_joint.shuffle(len(t_x)).batch(2).repeat(3)
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# バッチ → シャッフル → リピート
ds = ds_joint.batch(2).shuffle(len(t_x)).repeat(3)
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# バッチ → リピート → シャッフル
ds = ds_joint.batch(2).repeat(3).shuffle(len(t_x))
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())
