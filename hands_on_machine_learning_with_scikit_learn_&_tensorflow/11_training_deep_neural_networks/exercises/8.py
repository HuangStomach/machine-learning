import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train = X_train_full[5000:]
y_train = y_train_full[5000:]
X_valid = X_train_full[:5000]
y_valid = y_train_full[:5000]

model = keras.models.Sequential()
model.add(layers.Flatten(input_shape=[32, 32, 3]))
model.add(layers.BatchNormalization())

for _i in range(20):
    model.add(layers.Dense(100, kernel_initializer="lecun_normal", activation="selu"))
model.add(layers.AlphaDropout(rate=0.1))
model.add(layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.SGD(lr=1e-2)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
callbacks = [early_stopping_cb]

X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)
X_train_scaled = (X_train - X_means) / X_stds
X_valid_scaled = (X_valid - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds

model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_valid_scaled, y_valid), callbacks=callbacks)
