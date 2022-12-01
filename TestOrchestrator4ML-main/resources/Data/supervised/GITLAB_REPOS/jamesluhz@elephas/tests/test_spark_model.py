from __future__ import absolute_import
from __future__ import print_function
import pytest

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd


# Define basic parameters
batch_size = 64
nb_classes = 10
epochs = 1

# Create Spark context
pytest.mark.usefixtures("spark_context")


# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer="sgd",
              loss="categorical_crossentropy", metrics=["acc"])


def test_spark_model_end_to_end(spark_context):
    rdd = to_simple_rdd(spark_context, x_train, y_train)

    # sync epoch
    spark_model = SparkModel(model, frequency='epoch',
                             mode='synchronous', num_workers=2)
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=2, validation_split=0.1)
    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', score[1])

    # sync batch
    spark_model = SparkModel(model, frequency='batch',
                             mode='synchronous', num_workers=2)
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=2, validation_split=0.1)
    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', score[1])

    # async epoch
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=2, validation_split=0.1)
    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', score[1])

    # hog wild epoch
    spark_model = SparkModel(model, frequency='epoch', mode='hogwild')
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=2, validation_split=0.1)
    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', score[1])
