import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.datasets.mnist as mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.SGD(lr=3e-1),
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
model.evaluate(X_train, y_train)
