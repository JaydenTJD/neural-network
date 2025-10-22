import keras
from keras.datasets import mnist
import numpy as np

(train_data, train_answers), (test_data, test_answers) = mnist.load_data()

train_data = train_data.reshape(60000, 784)
test_data = test_data.reshape(10000, 784)
train_data = train_data.astype('float32') / 255
test_data = test_data.astype('float32') / 255
train_data = np.where(train_data >= 0.5, 1, 0).astype('int8')  # (60000 x 784) binary values
test_data = np.where(test_data >= 0.5, 1, 0).astype('int8')  # (10000 x 784) binary values
train_answers = keras.utils.to_categorical(train_answers, 10)  # (60000 x 10) 1-hot encoded
test_answers = keras.utils.to_categorical(test_answers, 10)  # (10000 x 10) 1-hot encoded

model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        keras.layers.Dense(32),
        keras.layers.Activation("relu"),
        keras.layers.Dense(32),
        keras.layers.Activation("relu"),
        keras.layers.Dense(10),
        keras.layers.Activation("softmax")
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_data, train_answers, batch_size=200, epochs=15, validation_split=0.0)
loss_and_metrics = model.evaluate(test_data, test_answers)
print("Test loss", loss_and_metrics[0])
print("Test accuracy", loss_and_metrics[1])
print(loss_and_metrics)