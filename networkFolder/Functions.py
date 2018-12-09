from typing import List
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


# loss functions
class LossFuncAbstract:
    @staticmethod
    def func(guess: List[float], answer: List[float]):
        pass

    @staticmethod
    def derivative(guess: List[float], answer: List[float], index: int):
        pass


class DiffSquare(LossFuncAbstract):
    @staticmethod
    def func(guess: List[float], answer: List[float]) -> float:
        return (np.linalg.norm(np.subtract(answer, guess))) ** 2

    # ∂q/∂yLi L - last layer index, i - index of output neuron
    @staticmethod
    def derivative(guess: List[float], answer: List[float], index: int) -> float:
        return 2*(answer[index] - guess[index])
