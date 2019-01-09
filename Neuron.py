from typing import List, Union, Type, Tuple
import numpy as np
import Functions
import inspect


class Neuron:
    """
    Represents single neuron. Includes all operations it does.

    Attributes
    ----------
    weights : List[float]
        vector of values that appropriate values from input are multiplied by (before they are summed)
    activationFunction : Type[Functions.FuncAbstract]
        an activation function that neuron uses
    """

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

    def process_input(self, input_vector: List[float]) -> Tuple[float, float]:
        """
        Makes neuron process given input
        :param input_vector: Input data
        :return: Pair of values - first is actual neuron output, second is calculated sum
        """
        input_vector_copy: List[float] = input_vector.copy()
        summ: float = np.dot(input_vector_copy, self.weights)
        result_tuple = (self.activationFunction.func(summ), summ)
        return result_tuple

    def adjust_weights(self, adjust_vector: List[float]):
        """Adjusts weights of neuron by values given in vector"""
        self.weights = np.add(self.weights, adjust_vector)
