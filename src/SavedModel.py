import os
import tensorflow as tf
import numpy as np
import pickle


def get_model():
    file_path = os.getcwd() + "/data/ocr_model.h5"
    return tf.keras.models.load_model(file_path)

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = get_model()
    model.evaluate(x_test, y_test)
    model.predict(x_test[0].reshape(1, 28, 28, 1))

    with open(os.getcwd() + "/data/save.pickle", "rb") as f:
        new_data = pickle.load(f)
        new_data = np.array(new_data).reshape(1, 28, 28, 1)
        f.close()

    print(np.ndim(x_test))
    print(np.ndim(new_data))
    predictions = model.predict(new_data)
    print(predictions)
