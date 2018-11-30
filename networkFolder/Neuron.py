from typing import List, Callable, Union
import numpy as np


def sigmoid(x) -> float:
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x) -> float:
    return x*(1-x)


def identity(x) -> float:
    return x


class Neuron:
    """Single neuron class. Includes all operations it does."""
    def __init__(self, init_weights: List[float], activation_function: Union[Callable, None]):
        """Initializes weight list and activation function of neuron."""
        assert type(init_weights) is list, "initWeights has to be type of list, but it's %s" % type(init_weights)
        assert (callable(activation_function) or activation_function is None),\
            "activationFunction has to be type of function or None, but it's %s" % type(activation_function)

        self.weights: List[float] = init_weights.copy()
        if activation_function is None:
            self.activationFunction: Callable = identity
        else:
            self.activationFunction: Callable = activation_function

    def process_input(self, input_vector: List[float]) -> float:
        """Makes neuron process given input."""
        # TODO do wywalenia bo siec bedzie to sprawdzac
        assert len(input_vector) == len(self.weights)-1,\
            "neuron %s requires input of size %d, got %d" % (self, len(self.weights)-1, len(input_vector))
        input_vector_copy: List[float] = input_vector.copy()
        input_vector_copy.append(1)
        result: float = np.dot(input_vector_copy, self.weights)
        return self.activationFunction(result)

    def adjust_weights(self, param):  # TODO
        # TODO
        pass

    def test(self, m):  # TODO do wywalenia
        return self.activationFunction(m)
