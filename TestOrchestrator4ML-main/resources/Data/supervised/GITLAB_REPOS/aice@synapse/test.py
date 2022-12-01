import tensorflow as tf
from tensorflow.keras import Model
# from synapse.network.classification.mnist import Mnist
from synapse.network.utilities.layers import Layers
from synapse.network.utilities.activations import Activations

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.network.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])


class Mnist(Model):
    def __init__(self):
        super(Mnist, self).__init__()

        self.dense1 = Layers.Dense(100, activation=Activations.ReLU)
        self.dense2 = Layers.Dense(50, activation=Activations.ReLU)
        self.dropout = Layers.Dropout(0.2)
        self.softmax = Layers.Softmax()

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        if training:
            x = self.dropout(x)
        x = self.softmax(x)

        return x

model = Mnist()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

