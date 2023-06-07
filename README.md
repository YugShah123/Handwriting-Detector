import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

for x in range(1, 6):
    img = cv.imread(f"C:/Users/shahy/PycharmProjects/HandwritingDetection/{x}.png", cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {x}.png")
        continue
    img = cv.resize(img, (28, 28))
    img = np.invert(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    print(f"The result is: {predicted_digit}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.show()
