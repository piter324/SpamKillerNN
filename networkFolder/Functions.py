import numpy as np


# TODO add other functions (ReLU, _/ , _/- )
class FuncAbstract:
    @staticmethod
    def func(x: float):
        pass

    @staticmethod
    def derivative(x: float):
        pass


class Sigmoid(FuncAbstract):
    @staticmethod
    def func(x: float) -> float:
        return 1/(1+np.exp(-x))

    @staticmethod
    def derivative(x: float) -> float:
        return x * (1 - x)


class Identity(FuncAbstract):
    @staticmethod
    def func(x: float) -> float:
        return x

    @staticmethod
    def derivative(x: float) -> float:
        return 1
