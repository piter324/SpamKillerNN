import typing

import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoidDerivative(x):
    return x*(1-x)


def identity(x):
    return x


class Neuron:
    """Single neuron class. Includes all operations it does."""
    def __init__(self, initWeights: list, activationFunction: typing.Callable):
        """Initializes weight list and activation function of neuron."""
        assert type(initWeights) is list, "initWeights has to be type of list, but it's %s"%type(initWeights)
        assert (callable(activationFunction) or activationFunction is None),\
            "activationFunction has to be type of function or None, but it's %s"%type(activationFunction)

        self.weights: list = initWeights.copy()
        if activationFunction is None:
            self.activationFunction: typing.Callable = identity
        else:
            self.activationFunction: typing.Callable = activationFunction

    def processInput(self, inputMatrix: list):
        """Makes neuron process given input."""
        inputMatrixCopy: list = inputMatrix.copy()
        inputMatrixCopy.append(1)
        print(inputMatrixCopy)
        result: int = np.dot(inputMatrixCopy, self.weights)
        return self.activationFunction(result)

    def test(self,m):
        return self.activationFunction(m)
