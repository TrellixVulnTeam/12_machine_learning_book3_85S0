'''Tensorflowで実際にモデル構築'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#%% データセットの作成
X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])
plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#%% 標準化し、Datasetオブジェクトを作成する
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)

ds_train_orig = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_train_norm, tf.float32),
    tf.cast(y_train, tf.float32)))

#%% モデル構築のためのクラスを作成(練習のため)
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(0.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')

    def call(self, x):
        return self.w*x + self.b

#%% インスタンスを作成し、概要を表示
model = MyModel()

model.build(input_shape=(None, 1))
model.summary()

#%% コスト関数の定義
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 関数の挙動テスト
yt = tf.convert_to_tensor([1.0])
yp = tf.convert_to_tensor([1.5])
loss_fn(yt, yp)

#%% 確率的勾配降下法の実装
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
    dW, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

#%% ハイパーパラメータを設定し、モデルを訓練
tf.random.set_seed(1)

# ハイパーパラメータ
num_epoch = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
steps_per_epoch = int(np.ceil(len(y_train) / batch_size))

# 訓練データの準備
ds_train = ds_train_orig.shuffle(buffer_size=len(y_train))
ds_train = ds_train.repeat(count=None)
ds_train = ds_train.batch(1)

# モデルの訓練
Ws, bs = [], []

for i, batch in enumerate(ds_train):
    if i >= steps_per_epoch * num_epochs:
        break
    Ws.append(model.w.numpy())
    bs.append(model.b.numpy())

    bx, by = batch
    loss_val = loss_fn(model(bx), by)

    train(model, bx, by, learning_rate=learning_rate)
    if i%log_steps==0:
        print('Epoch {:4d} Step {:2d} Loss {:6.4f}'.format(
            int(i/steps_per_epoch), i, loss_val))
