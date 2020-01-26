import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPool2D(3),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
model.evaluate(x_test, y_test)

file_path = os.getcwd() + "/data/ocr_model.h5"
model.save(file_path)

# Test accuracy of 0.9826
# Prediction usage
# y = model.predict(x_test[0].reshape(-1, 28, 28, 1))
# print(y)
# [[2.5660287e-07 2.0680563e-08 1.1196026e-05 4.9596492e-05 4.1060214e-10
#   5.3089999e-09 1.4606168e-13 9.9992871e-01 2.5187333e-07 1.0030492e-05]]
