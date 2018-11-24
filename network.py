import numpy as np

#Each row is a training example, each column is a feature
X = np.array((
    [0,0,1],
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,1]
), dtype=float)
y = np.array(([0],[1],[1],[1],[0]), dtype=float)

def sigmoid(t):
    return 1/(1+np.exp(-t))

def sigmoid_derivative(p):
    return p*(1-p)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],self.input.shape[0]) 
        self.weights2   = np.random.rand(self.input.shape[0],1)                 
        self.y          = y
        self.output     = None
        print("Weights 1: "+str(self.weights1))
        print("Weights 2: "+str(self.weights2))

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output)*sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2
    
    def train(self):
        self.output = self.feedforward()
        self.backprop()

NN = NeuralNetwork(X, y)

for i in range(1500): # trains the network 1500 times
    if i % 100 == 0:
        print("for interation #"+str(i)+"\n")
        print("Input: \n"+str(X))
        print("Actual output: \n"+str(y))
        print("Predicted output: \n"+str(NN.feedforward()))
        print("Loss: \n"+str(np.mean(np.square(y - NN.feedforward())))) # mean sum suared loss
        print("\n")
    
    NN.train()