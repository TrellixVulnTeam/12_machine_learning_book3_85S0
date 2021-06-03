'''Irisデータセットを分類する多層パーセプトロンを構築'''
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/12_machine_learning_book3/ch13
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


#%% irisデータセットの準備
iris, iris_info = tfds.load('iris', with_info=True)
print(iris_info)

#%% データセットの分割
tf.random.set_seed(1)
ds_orig = iris['train']
ds_orig = ds_orig.shuffle(150, reshuffle_each_iteration=False)
print(next(iter(ds_orig)))

ds_train_orig = ds_orig.take(100)
ds_test = ds_orig.skip(100)

# 分割の確認
n = 0
for example in ds_train_orig:
    n += 1
print(n)

n = 0
for example in ds_test:
    n += 1
print(n)

# タプルに変換
ds_train_orig = ds_train_orig.map(
    lambda x: (x['features'], x['label']))
ds_test = ds_test.map(
    lambda x: (x['features'], x['label']))
next(iter(ds_train_orig))

#%% モデルの作成
iris_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='sigmoid', name='fc1', input_shape=(4, )),
    tf.keras.layers.Dense(3, name='fc2', activation='softmax')])

iris_model.summary()

#%% compile
iris_model.compile(optimizer='adam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics=['accuracy'])

#%% 訓練
num_epochs = 100
training_size = 100
batch_size = 2
steps_per_epoch = np.ceil(training_size / batch_size)

ds_train = ds_train_orig.shuffle(buffer_size=training_size)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(buffer_size=1000)

history = iris_model.fit(ds_train, epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        verbose=0)

#%% 可視化
hist = history.history

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(hist['loss'], lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(hist['accuracy'], lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()

plt.show()


#%% 訓練したモデルをテストデータで評価する
results = iris_model.evaluate(ds_test.batch(50), verbose=0)
print('Test loss:{:.4f} Test Acc.:{:.4f}'.format(*results))

#%% 訓練したモデルの保存と読み込み
iris_model.save('iris-classifier.h5', overwrite=True,
                include_optimizer=True,
                save_format='h5')

iris_model_new = tf.keras.models.load_model('iris-classifier.h5')
iris_model_new.summary()
