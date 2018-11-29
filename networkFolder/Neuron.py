from typing import List, Callable
import numpy as np


def sigmoid(x) -> float:
    return 1/(1+np.exp(-x))


def sigmoidDerivative(x) -> float:
    return x*(1-x)


def identity(x) -> float:
    return x


class Neuron:
    """Single neuron class. Includes all operations it does."""
    def __init__(self, initWeights: List[float], activationFunction: Callable):
        """Initializes weight list and activation function of neuron."""
        assert type(initWeights) is list, "initWeights has to be type of list, but it's %s"%type(initWeights)
        assert (callable(activationFunction) or activationFunction is None),\
            "activationFunction has to be type of function or None, but it's %s"%type(activationFunction)

        self.weights: List[float] = initWeights.copy()
        if activationFunction is None:
            self.activationFunction: Callable = identity
        else:
            self.activationFunction: Callable = activationFunction

    def processInput(self, inputMatrix: List[float]) -> float:
        """Makes neuron process given input."""
        assert len(inputMatrix) == len(self.weights)-1
        inputMatrixCopy: List[float] = inputMatrix.copy()
        inputMatrixCopy.append(1)
        #print(inputMatrixCopy)
        result: float = np.dot(inputMatrixCopy, self.weights)
        return self.activationFunction(result)

    def test(self,m):
        return self.activationFunction(m)
