import mnist_loader
import network


""" Neural network pattern recognition
    algorithm experiment using sigmoidal network and
    MNIST collection of hand-written letters"""

# Get the MNIST data
print("loading MNIST data:")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 784 = 28 x 28 (letter dimension)
# 30 (hidden layer size)
# 10 (output size)
# Construct a sigmoidal net
print("creating neural net")
net = network.Network([784, 30, 10])

print("gradient descent minimisation learning:")
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)