import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

basewidth = 28
baseheight = 28

img = Image.open('1.png')

img = img.resize((28, 28))
img.save("processed_image.png")
img = Image.open("processed_image.png").convert("LA")
img.save("processed_image.png")
print("done processing image")

image = Image.open("processed_image.png")
data = np.asarray(image).reshape(-1, 28, 28, 1)
print(type(data))
print(data)
print(data.shape)


model = tf.keras.models.load_model("digit_recognizer_.h5")

print(model.predict(data)[0])


