from typing import List, Union, Type
import numpy as np
import Functions
import inspect


class Neuron:
    """Single neuron class. Includes all operations it does."""
    def __init__(self, init_weights: List[float], activation_function: Union[Type[Functions.FuncAbstract], None]):
        """Initializes weight list and activation function of neuron."""
        assert type(init_weights) is list, "initWeights has to be type of list, but it's %s" % type(init_weights)
        assert (inspect.isclass(activation_function) or activation_function is None),\
            "activationFunction has to be type of function or None, but it's %s" % type(activation_function)

        self.weights: List[float] = init_weights.copy()
        if activation_function is None:
            self.activationFunction: Type[Functions.FuncAbstract] = Functions.Identity
        else:
            self.activationFunction: Type[Functions.FuncAbstract] = activation_function

    def process_input(self, input_vector: List[float]) -> float:
        """Makes neuron process given input."""
        # TODO do wywalenia bo siec bedzie to sprawdzac
        assert len(input_vector) == len(self.weights)-1,\
            "neuron %s requires input of size %d, got %d" % (self, len(self.weights)-1, len(input_vector))
        input_vector_copy: List[float] = input_vector.copy()
        input_vector_copy.append(1)
        result: float = np.dot(input_vector_copy, self.weights)
        return self.activationFunction.func(result)

    def adjust_weights(self, adjust_vector: List[float]):
        self.weights = np.subtract(self.weights, adjust_vector)
