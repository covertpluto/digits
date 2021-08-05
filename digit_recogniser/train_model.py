import numpy
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import load_dataset

x_train, y_train, x_test = load_dataset.get_data()
print(x_train.shape)
x_train = x_train.values.reshape((-1, 28, 28, 1))

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Flatten())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    epochs=30,
                    verbose=1,
                    validation_split=0.2)

plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_accuracy"])
plt.plot(history.history["val_loss"])
plt.legend(["accuracy", "loss", "val_accuracy", "val_loss"])
plt.show()
model.save("./digit_recognizer_.h5")
