import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pickle
from tqdm import tqdm

from sklearn.datasets import fetch_openml
mnist = fetch_openml(name='mnist_784')

print(mnist.keys())

data = mnist.data
labels = mnist.target

# n stores a random index from the dataset
# data.shape[0] gives the number of samples
# from 0 upto and including the number of rows in the dataset 
n = np.random.choice(np.arange(data.shape[0]+1))
print(n)

# retrieves one specific image from the dataset using the random index n
# .iloc is a pandas indexing method that selects data based on integer position
# after selecting the row, .values converts the pandas series into a NumPy array, this is done to make math operations like reshaping easier
test_img = data.iloc[n].values

# extracts the corresponding label for the selected image
# In the MNIST dataset, labels are digits from 0 to 9, indicating which handwritten digit the image represents
# mnist.target is a pandas Series containing the labels for each image
test_label = mnist.target.iloc[n]

print(test_img.shape)

# calculates the side length of the image
# MNIST images are 28x28 pixels, so the side length is 28
# but they are stored as a flat array of 784 pixels
# taking sqrt of this and convering to int gives the original side length of 28
side_length = int(np.sqrt(test_img.shape))

# transforms the One-dimensional array into a two-dimensional matrix 
# this represents the image in its original 2D form.
# this is necessary because images are inherently 2D structures
# Neural networks typically expect input data in a structured format and they process flattened (1D) data
reshaped_test_img = test_img.reshape(side_length, side_length)

print("Image label: ", str(test_label))
plt.imshow(reshaped_test_img, cmap='gray') #greyscale render of image
plt.axis('off')
plt.show()

# Rule for the shape of the weight matrix:
# If Layer Ln has Sn nodes and layer Ln+1 has Sn+1 nodes, then the On matrix of parameters connecting Ln to Ln+1 will  have the shape (Sn, Sn+1).

w1 = np.ones((784,4)) * 0.01
z1 = np.dot(data, w1)  # data is (70000, 784), w1 is (784, 4), so z1 will be (70000, 4)
print (z1.shape)
w2 = np.ones((4,10))
z2 = np.dot(z1, w2)  # z1 is (70000, 4), w2 is (4, 10), so z2 will be (70000, 10)
print (z2.shape)



## Activation functions

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0,z)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, z, z * 0.01)

def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z))
    return e / np.sum(e, axis=0)

# Scales the input data to the [0,1] range
def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# turns the array of labels from an n-sized vector (n is the number of samples)
# to an n x m array ( m is the number of possible outputs)
def one_hot_encode(x: np.ndarray, num_labels: int) -> np.ndarray:
    return np.eye(num_labels)[x]

# Derivative of the activation functions to perform gradient descent
def derivative(function_name: str, z: np.ndarray) -> np.ndarray:
    if function_name == "sigmoid":
        return sigmoid(z) * (1 - sigmoid(z))
    if function_name == "tanh":
        return 1 - np.square(tanh(z))
    if function_name == "relu":
        y = (z > 0) * 1
        return y
    if function_name == "leaky_relu":
        return np.where(z > 0, 1, 0.01)
    return "No such activation function"


## Implementing the Neural Network
class NN(object):
    def __init__(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, activation: str, num_labels: int, architecture: List[int]):
        # Normalise the training data in range of 0 to 1
        self.X = normalize(X)
        # assertion is performed to verify that the normalisation worked correctly
        assert np.all((self.X >= 0) & (self.X <= 1))

        # create copies of the input data to avoid modifying the original data
        self.X, self.X_test = X.copy(), X_test.copy()
        self.y, self.y_test =y.copy(), y_test.copy()

        # define the dictionary to store the results of activation during forward propagation
        self.layers = {}
        # the size of the hidden layers is stored as an array; defines the network structure
        self.architecture = architecture
        # stores the activation function name
        self.activation = activation

        assert self.activation in ["relu", "sigmoid", "tanh", "leaky_relu"]
        # stores weights and biases for each layer
        self.parameters = {}
        self.num_labels = num_labels
        self.m = X.shape[1]

        # architecture list is modified to include the input size at the beginning
        # and the output size at the end
        self.architecture.append(self.num_labels)
        self.num_input_features = X.shape[0]
        self.architecture.insert(0, self.num_input_features)
        self.L = len(architecture)
        # Assertions to verify that the data dimesnsions match expectations
        # Ensuring X has the shape (num_input_features, m) and y has the shape (num_labels, m)
        # where m is the number of samples
        assert self.X.shape == (self.num_input_features, self.m)
        assert self.y.shape == (self.num_labels, self.m)

    # Initializes the parameters (weights and biases) for each layer in the neural network, except the input layer
    def initialize_parameters(self):
        for i in range(1, self.L):
            print(f"Initializing parameters for layer: {i}.")
            # creates a matrix of random numbers drawn from a standard normal distribution (mean 0, variance 1)
            # scales these values by 0.01 to keep initial weights small, which helps prevents exploding gradients
            # the dimensions are set to match the number of nodes in the current layer and the previous layer
            # each neuron in the current layer gets one weight for each neuron in the previous layer
            # stores the weights in the parameters dictionary with keys "w1", "w2 
            self.parameters["w" + str(i)] = np.random.randn(self.architecture[i], self.architecture[i-1]) * 0.01
            # creates a column vector of zeroes with dimensions matching the number of neurons in the current layer
            # uses zeroes initialization for biases as a practice
            # stores the biases in the parameters dictionary with keys "b1", "b2", etc.
            self.parameters["b" + str(i)] = np.zeros((self.architecture[i], 1))

# Feedforward

    def forward(self):
        params = self.parameters
        self.layers["a0"] = self.X
        # iterates through the hidden layers of the network except the input and output layer.
        for l in range(1, self.L - 1):
            # Linear Transformation: performs the weighted sum plus bias.
            self.layers["z" + str(1)] = np.dot(params["w" + str(l),
                                                      self.layers["a" + str(l - 1)]]) + params["b" + str(l)]
            # Activation: appliies the softmax activation. 
            # The eval function dynamically calls the appropriate activation function based on the string stored in self.activation.
            self.layers["a" + str(l)] = eval(self.activation)(self.layers["z" + str(l)])
            assert self.layers["a" + str(l)].shape == (self.architecture[l], self.m)
        # Linear Transformation for the output layer.
        self.layers["z" + str(self.L-1)] = np.dot(params["w" + str(self.L-1)],
                                                  self.layers["a" + str(self.L-2)]) + params["b" + str(self.L-1)]
        # Applies Softmax Activation specifically to the output layer, appropriate for classification tasks.
        self.layers["a" + str(self.L-1)] = softmax(self.layers["z" + str(self.L-1)])
        # The result is stored in self.output for easy access.
        self.output = self.layers["a" + str(self.L-1)]
        # The assertion checks that the output shape matches expectations.
        assert self.output.shape == (self.num_labels, self.m)
        # All columns sum to 1 ( a property of softmax outputs)
        assert all([s for s in np.sum(self.output, axis = 1)])

        # The cross-entropy cost is calculated with this.
        # The small costant prevents taking the logarithm of zero.
        cost = - np.sum(self.y * np.log(self.output + 0.000000001))

        # The method returns both the cost and all layer activations.
        return cost, self.layers


# Backpropagation

    def backpropagate(self):
        # empty dictionary to store gradients.
        derivatives = {}

        # calculates the output layer error.
        # this is the gradient of the cost function with respect to the final layer activations.
        dZ = self.output - self.y

        # checks the shape of the layer error.
        assert dZ.shape == (self.num_labels, self.m)

        # weight gradient calculation for the output layer.
        # multiplies the output error by the transpose of the previous layer's activations. .
        # normalised by the number of examples .
        dW = np.dot(dZ, self.layers["a" + str(self.L-2)].T) / self.m

        # bias gradient calculation for the output layer.
        # averages the error across all examples.
        db = np.sum(dZ, axis=1, keepdims=True) / self.m

        # gradient calculation to propagate to the previous layer.
        dAPrev = np.dot(self.parameters["w" + str(self.L-1)].T, dZ)

        # the gradients are stored in the derivatives dictionary 
        derivatives["dW" + str(self.L-1)] = dW
        derivatives["db" + str(self.L-1)] = db

        # we iterate backwards through the hidden layers 
        for l in range(self.L-2, 0, -1):
            # calculates the gradients for the error at the current layer
            # it combines the error from the next layer with the derivative of the current layer's activation function
            dZ = dAPrev * derivative(self.activation, self.layers["z" + str(l)])

            # calculates the weight gradients 
            dW = 1. / self.m * np.dot(dZ, self.layers["a" + str(l-1)].T)

            # calculates the bias gradients
            db = 1. / self.m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                # calculates the gradient for the  previous layer
                # (if not at the first hidden layer)
                dAPrev = np.dot(self.parameters["w" + str(l)].T, (dZ))
            # all gradients are stored in the derivatives dictionary 
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
        
        # the helper function calculates the derivatives of various activation functions based on their name.
        self.derivatives = derivatives

        return self.derivatives  
    
    

# Fitting, accuracy and predictions

    def fit(self, lr=0.01, epochs = 1000):
        self.costs = []
        self.initialize_parameters()
        self.accuracies = {"train": [], "test": []}
        for epoch in tqdm(range(epochs), colour="BLUE"):
            cost, cache = self.forward()
            self.costs.append(cost)
            derivatives = self.backpropagate()
            for layer in range(1, self.L):
                self.parameters["w"+str(layer)] = self.parameters["w"+str(layer)] - lr * derivatives["dW" + str(layer)]
                self.parameters["b"+str(layer)] = self.parameters["b"+str(layer)] - lr * derivatives["db" + str(layer)]            
            train_accuracy = self.accuracy(self.x, self.y)
            test_accuracy = self.accuracy(self.X_test, self.y_test)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch:3d} | Cost: {cost:.3f} | Accuracy: {train_accuracy:.3f}")
            self.accuracies["train"].append(train_accuracy)
            self.accuracies["test"].append(test_accuracy)
        print("Training Terminated")


    def predict(self,x):
        params = self.parameters
        n_layers = self.L - 1
        values = [x]
        for l in range(1, n_layers):
            z = np.dot(params["w" + str(l)], values[l-1]) + params["b" + str(l)]
            a = eval(self.activation)(z)
            values.append(a)
        z = np.dot(params["w" + str(n_layers)], values[n_layers - 1]) + params["b" + str(n_layers)]
        a = softmax(z)
        if x.shape[1] > 1:
            ans = np.argmax(a, axis = 0)
        else:
            ans = np.argmax(a)
        return ans
    

    def accuracy(self, x, y):
        P = self.predict(x)
        return sum(np.equal(P, np.argmax(y, axis = 0))) / y.shape[1] * 100
    
    def pickle_model(self, name: str):
        with open("fitted model_" + name + ".pickle", "wb") as modelFile:
            pickle.dump(self, modelFile)

    def plot_counts(self):
        counts = np.unique(np.argmax(self.output, axis = 0), return_counts = True)
        plt.bar(counts[0], counts[1], color = "navy")
        plt.ylabel("Counts")
        plt.xlabel("y_hat")
        plt.title("Distribution of predictions")
        plt.show()

    
    def plot_cost(self, lr):
        plt.figure(figsize = (8,4))
        plt.plot(np.arange(0, len(self.costs)), self.costs, lw = 1, color = "orange")
        plt.title(f"Learning rate: {lr}\nFinal Cost: {self.costs[-1]:.5f}", fontdict = {
            "family":"sans-serif",
            "size": "12"})
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()

    
    def plot_accuracies(self, lr):
        acc = self.accuracies
        fig = plt.figure(figsize = (6,4))
        ax = fig.add_subplot(111)
        ax.plot(acc["train"], label = "train")
        ax.plot(acc["test"], label = "test")
        plt.legend(loc = "lower right")
        ax.set_title("Accuracy")
        


