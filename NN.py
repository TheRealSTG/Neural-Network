import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pickle
from tqdm import tqdm
from tabulate import tabulate

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
side_length = int(np.sqrt(test_img.shape[0]))

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
    # Subtract max for numerical stability
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

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
        return 1 - np.power(tanh(z), 2)
    if function_name == "relu":
        return np.where(z > 0, 1, 0)
    if function_name == "leaky_relu":
        return np.where(z > 0, 1, 0.01)
    return "No such activation function"


## Implementing the Neural Network
class NN(object):
    def __init__(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, activation: str, num_labels: int, architecture: List[int]):
        # Create copies first, then normalize
        self.X, self.X_test = X.copy(), X_test.copy()
        self.X = normalize(self.X)
        self.X_test = normalize(self.X_test)  # Also normalize test data

        # Add these two lines to store y and y_test
        self.y = y
        self.y_test = y_test

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

        self.costs = []  # Add this line - you're using it in fit() but never initialized it
        self.train_accuracies = []  # Add this line
        self.test_accuracies = []  # Add this line
        self.initialize_parameters()  # Add this line

    # Initializes the parameters (weights and biases) for each layer in the neural network, except the input layer
    def initialize_parameters(self):
        # Use architecture as is, without adding dimensions again
        for l in range(1, self.L):
            # Different initialization based on activation function
            if self.activation == "relu" or self.activation == "leaky_relu":
                # He initialization for ReLU variants
                self.parameters[f"W{l}"] = np.random.randn(self.architecture[l], self.architecture[l-1]) * np.sqrt(2./self.architecture[l-1])
            else:
                # Xavier/Glorot initialization for sigmoid and tanh
                self.parameters[f"W{l}"] = np.random.randn(self.architecture[l], self.architecture[l-1]) * np.sqrt(1./self.architecture[l-1])
            
            self.parameters[f"b{l}"] = np.zeros((self.architecture[l], 1))

# Feedforward

    def forward(self):
        cache = {}
        A = self.X  # Input features
        L = len(self.parameters) // 2  # Number of layers
        
        # Forward propagation
        for l in range(1, L+1):
            Z = np.dot(self.parameters[f"W{l}"], A) + self.parameters[f"b{l}"]
            cache[f"Z{l}"] = Z
            
            # Apply activation function (except for output layer)
            if l == L:  # Output layer
                A = softmax(Z)
            else:  # Hidden layers
                if self.activation == "sigmoid":
                    A = sigmoid(Z)
                elif self.activation == "tanh":
                    A = tanh(Z)
                elif self.activation == "relu":
                    A = relu(Z)
                elif self.activation == "leaky_relu":
                    A = leaky_relu(Z)
            
            cache[f"A{l}"] = A
        
        # Compute cost
        m = self.y.shape[1]
        cost = -np.sum(self.y * np.log(A + 1e-8)) / m  # Add small epsilon for numerical stability
        
        return cost, cache


# Backpropagation

    def backpropagate(self, cache):
        derivatives = {}
        m = self.y.shape[1]
        L = len(self.parameters) // 2  # Number of layers
        
        # Output layer error (using softmax derivative)
        A_output = cache[f"A{L}"]  # Final activation (softmax output)
        dZ = A_output - self.y  # Derivative of cross-entropy loss with softmax
        
        # Gradient for output layer
        derivatives[f"dW{L}"] = (1/m) * np.dot(dZ, cache[f"A{L-1}"].T)
        derivatives[f"db{L}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Hidden layers
        for l in reversed(range(1, L)):
            # Backpropagate the error
            dA = np.dot(self.parameters[f"W{l+1}"].T, dZ)
            
            # Apply derivative of activation function
            if self.activation == "sigmoid":
                dZ = dA * derivative("sigmoid", cache[f"Z{l}"])
            elif self.activation == "tanh":
                dZ = dA * derivative("tanh", cache[f"Z{l}"])
            elif self.activation == "relu":
                dZ = dA * derivative("relu", cache[f"Z{l}"])
            elif self.activation == "leaky_relu":
                dZ = dA * derivative("leaky_relu", cache[f"Z{l}"])
            
            # Calculate gradients
            if l > 0:
                prev_A = cache[f"A{l-1}"] if l > 1 else self.X
                derivatives[f"dW{l}"] = (1/m) * np.dot(dZ, prev_A.T)
                derivatives[f"db{l}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        return derivatives
    

# Fitting, accuracy and predictions

    # implements the standart gradeient descent training algorithm for neural networks.
    # for each epoch, the method performs a forward pass by calling self.forward() which computes the correct cost (loss)
    # it caches intermediate values needed for backpropagation.
    # the cost is then appended to a list for tracking training progess.
    # iteratively adjusts network parameters to minimize the loss function over a specified number of epochs.
    # the accuracy is only calculated and printed every 10 epochs, which helps reduce computational overhead and clutter in the output.
    def fit(self, lr=0.01, epochs=100):
        # provides a progress bar for a visual feedback during training.
        for i in tqdm(range(epochs)):
            # Forward pass
            cost, cache = self.forward()
            self.costs.append(cost)
            
            # Backpropagation
            # computes the gradients of the cost with respect to the network's weights and biases.
            # these gradients are used to update the parameters for each layer.
            # the weights and biases are adjusted by subtracting the product of the learning rate and their respective gradients. 
            # this step moves the parameters in the direction that reduces the loss.
            derivatives = self.backpropagate(cache)
            
            # Update parameters
            for l in range(1, len(self.parameters) // 2 + 1):
                self.parameters[f"W{l}"] -= lr * derivatives[f"dW{l}"]
                self.parameters[f"b{l}"] -= lr * derivatives[f"db{l}"]
            
            # Calculate and store accuracies
            # for every 10 epochs, the method evaluates the model's accuracy on bot the training and test datasets using the accuracy method.
            # these values are stored for later analysis and printed to provide feedback on the training progress.
            if i % 10 == 0:
                train_accuracy = self.accuracy(self.X, self.y)
                test_accuracy = self.accuracy(self.X_test, self.y_test)
                self.train_accuracies.append(train_accuracy)
                self.test_accuracies.append(test_accuracy)
                print(f"Epoch {i}/{epochs}, Cost: {cost:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
        print("Training Terminated")


    # implements the forward propagation process for making predictions with a trained neural network.
    def predict(self, x):
        # retrieves the network parameters
        params = self.parameters
        # determines the number of layers
        n_layers = self.L - 1
        # initialises the values list with the input data as the first element    
        values = [x]
        # to propagate the input through each hidden layer
        # for each hidden layer in the network
        for l in range(1, n_layers):
            # Use uppercase "W" to match initialization
            z = np.dot(params["W" + str(l)], values[l-1]) + params["b" + str(l)]
            
            # applies the activation function to introduce non-linearity
            a = eval(self.activation)(z)
            values.append(a)
        # Also use uppercase "W" here
        z = np.dot(params["W" + str(n_layers)], values[n_layers - 1]) + params["b" + str(n_layers)]

        # the softmax function applied to the activation function
        a = softmax(z)

        # determines the predicted class
        if x.shape[1] > 1:
            ans = np.argmax(a, axis = 0)
        else:
            ans = np.argmax(a)
        return ans
    

    # the method evaluates the model's performance by comparing predictions agains ground truth labels.
    def accuracy(self, x, y):
        # calls predict(x) to get predicted class indices
        # compares these predictions with the true labels
        # it converts one-hot encoded labels to indices
        # calculates the percentage of correct predictions
        P = self.predict(x)
        return sum(np.equal(P, np.argmax(y, axis = 0))) / y.shape[1] * 100
    

    # the pcikle model method saves the trained model to disk using python's pickle module.
    # opens a file with a name based on the input parameter
    # serializes the entirte model object for later retieval 
    def pickle_model(self, name: str):
        with open("fitted_model_" + name + ".pickle", "wb") as modelFile:
            pickle.dump(self, modelFile)


    # creates a bar chart showing the distribution of predictions across classes
    # extracts unique prediction values and their counts
    # displays the frequency of each predicted class
    def plot_counts(self):
        # Run forward pass to get predictions
        _, cache = self.forward()
        output = cache[f"A{len(self.parameters) // 2}"]  # Get the final activation
        
        # Count predictions
        predictions = np.argmax(output, axis=0)
        unique, counts = np.unique(predictions, return_counts=True)
        
        plt.figure(figsize=(8, 4))
        plt.bar(unique, counts, color="navy")
        plt.ylabel("Counts")
        plt.xlabel("Predicted Class")
        plt.title("Distribution of Predictions")
        plt.xticks(unique)
        plt.show()

    # visualizes the cost/loss over training epochs
    # plots the cost values that are stored in self.costs
    # includes the learning rate and final cost in the title
    # useful for monitoring convergence and learning rate effectiveness
    def plot_cost(self, lr):
        plt.figure(figsize = (8,4))
        plt.plot(np.arange(0, len(self.costs)), self.costs, lw = 1, color = "orange")
        plt.title(f"Learning rate: {lr}\nFinal Cost: {self.costs[-1]:.5f}", fontdict = {
            "family":"sans-serif",
            "size": "12"})
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()

    
    # compares the training and testing acccuracy over time
    # plots both the accuracy curves on the same graph
    # includes a legend to distinguish between training and testing data
    # it helps identify overfitting (when training accuracy far exceeds testing accuracy)
    # final accuracy values for both training and test datasets are annotated.
    # top and right spines or borders of the plot are removed
    def plot_accuracies(self, lr):
        # Changed self.accuracies to use the actual attributes
        fig = plt.figure(figsize = (6,4))
        ax = fig.add_subplot(111)
        ax.plot(self.train_accuracies, label = "train")
        ax.plot(self.test_accuracies, label = "test")
        plt.legend(loc = "lower right")
        ax.set_title("Accuracy")
        
        # Make sure there are accuracy values before attempting to plot
        if len(self.train_accuracies) > 0:
            ax.annotate(f"Train: {self.train_accuracies[-1]: .2f}", 
                        (len(self.train_accuracies)+4, self.train_accuracies[-1]+2), color="blue")
        if len(self.test_accuracies) > 0:
            ax.annotate(f"Test: {self.test_accuracies[-1]: .2f}", 
                        (len(self.test_accuracies)+4, self.test_accuracies[-1]-2), color="orange")
        
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.show()

    def __str__(self):
        return str(self.architecture)
        
# Remove the attempt to load from a pickle file
# with open("mnist.pickle", 'rb') as f:
#     mnist = pickle.load(f)

# prepares the training and testing datasets

# determines the number of samples used for training.
# for the training set, the first 60,000 samples are selected and transposed(.T) so that each column represents a single sample,
# this is a common format for NN input.
train_test_split_no = 60000

# input features and labels are split into training and test sets based on the train_test_split_no
# the labels are also selected, converted to integers and then one hot encoded using the one_hot_encode function.
X_train = data.values[:train_test_split_no].T
y_train = labels[:train_test_split_no].values.astype(int)
# this function transforms each label into a vector of length 10 (since there are 10 possible classes)
# where only the index corresponding to the label is set to 1 and the rest are 0.
# the result is then transposed so that each column matches the format of the input features
y_train = one_hot_encode(y_train, 10).T

X_test = data.values[train_test_split_no:].T
y_test = labels[train_test_split_no:].values.astype(int)
y_test = one_hot_encode(y_test, 10).T

print(X_train.shape, X_test.shape)

# Define activation configurations with optimized hyperparameters
activation_configs = {
    "relu": {"lr": 0.003, "epochs": 200},
    "sigmoid": {"lr": 0.0005, "epochs": 300},
    "tanh": {"lr": 0.0007, "epochs": 300},
    "leaky_relu": {"lr": 0.0015, "epochs": 200}
}

# specfifies the structure of the hidden layers in the network, with two layers containing 128 and 32 neurons.
architecture = [128, 32]
# stores instances of models that have been trained, possibly with different activation functions or hyperparameters.
trained_models = []

# takes a list o fmodels and the training and test datasets as input.
# prints a formatted table comparing the train and test accuracies of each model.
# the accuracy method computes the %age of correct predictions by comparing the model's output with the true labels
# (converting one hot encoded labels to class indices as needed)
def print_accuracies(models, X_train, y_train, X_test, y_test):
    """Print a comparison table of model accuracies."""
    headers = ["Activation", "Train Accuracy (%)", "Test Accuracy (%)"]
    rows = []
    
    for model in models:
        train_acc = model.accuracy(X_train, y_train)
        test_acc = model.accuracy(X_test, y_test)
        rows.append([model.activation, f"{train_acc:.2f}", f"{test_acc:.2f}"])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

# Provides a visual to inspect how well a trained model performs.
# creates a gird of subplots using matplotlibs with each subplot displaying one test sample.
def plot_predictions(model, X_test, y_test, rows=2, cols=4):
    """Plot sample predictions from the model."""
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    
    # randomly chooses a set of indices from the test set,e nsuring the number of samples matches the grid size but does not exceed the available test data.
    n_samples = min(rows * cols, X_test.shape[1])
    indices = np.random.choice(X_test.shape[1], n_samples, replace=False)
    
    # for each selected sample, the function extracts the image data and its true label. 
    # the true label is determined by finding the index of the max value in the one-hot encoded label vector.
    for i, idx in enumerate(indices):
        sample = X_test[:, idx:idx+1]
        true_label = np.argmax(y_test[:, idx])
        
        # model's prediction for each sample is obtained here.
        # returns the predicted class index.
        pred_label = model.predict(sample)
        
        # sample is reshaped into a 28x28 image and displayed in greyscale.
        # each subplot is annotated with both the true and predicted lables, making it easy to spot correct and incorrect predictions visually.
        img = sample.reshape(28, 28).T
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {true_label}, Pred: {pred_label}")
        axes[i].axis('off')
    
    # used to minimize overlap and esnure that the labels, titles and axes are clearly visible.
    plt.tight_layout()
    plt.show()

# Train each model once with optimized parameters
# iterates over a dict of activation function configs
# tains a spearate NN for each activation type.
# for each entry it prints info about the activation function, learning ratem and number of epochs being used.
# it then constructs parameters needed to initliaize the NN including the training an dtesting data and activation function name, number of output classes(10) and a copy of the architecture list.
# the architecture is copied for each model to avoid unintended modifications due to Python's list mutability.

for activation, config in activation_configs.items():
    print(f"\n\n===== Training model with {activation} activation =====")
    print(f"Learning rate: {config['lr']}, Epochs: {config['epochs']}")
    
    params = [X_train, y_train, X_test, y_test, activation, 10, architecture.copy()]

    # a new instance of the NN class is created with these parameters.
    # the model is trained using the specified learning rate and number of epochs by calling its fit method.
    # this performs forward and backward passes, updates weights and tracks cost and accuracy over time.
    model = NN(*params)
    model.fit(lr=config['lr'], epochs=config['epochs'])
    
    # after training, the model is added to the list for comparison.
    trained_models.append(model)
    
   # plots the cost (loss) curve and the accuracy curves using the model's plot_cost and plot_accuracies methods.
    model.plot_cost(config['lr'])
    model.plot_accuracies(config['lr'])

# summarizes and compare accuracies of all models that are trained on different activation functions.
print("\n===== Activation Functions Comparison =====")
print_accuracies(trained_models, X_train, y_train, X_test, y_test)

# plot predictions for each model
print("\n===== Sample Predictions by Activation Function =====")
for model in trained_models:
    print(f"\nPredictions using {model.activation} activation:")
    plot_predictions(model, X_test, y_test, rows=2, cols=4)