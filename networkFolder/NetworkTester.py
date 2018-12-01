from NeuralNetwork import NeuralNetwork
from TrainingSet import TrainingSet
from typing import List, Tuple
import numpy as np


class NetworkTester:
    @staticmethod
    def calc_loss(guess: List[float], answer: List[float]) -> float:
        """Calculates loss with formula: q = ||answer - guess||^2"""
        print("Guess: %s" % guess)
        print("Answer: %s" % answer)
        return (np.linalg.norm(np.subtract(answer, guess))) ** 2

    @staticmethod
    def test(neural_network: NeuralNetwork, training_set: TrainingSet) -> Tuple[float, float]:
        # Checking if training_set is legal for neural_network
        assert len(training_set.data[0]) == len(neural_network.layers[0][0].weights)-1
        assert len(training_set.answers[0]) == len(neural_network.layers[len(neural_network.layers)-1])
        print("###STARTING TEST...###")
        # Target function (to minimize) J = (1/N) * Σ t=1 do N ||answer_t - guess_t||^2
        j_function: float = 0
        correct_counter: int = 0
        for pair_index in range(len(training_set.data)):
            single_result: List[float] = neural_network.guess(training_set.data[pair_index])
            loss: float = NetworkTester.calc_loss(single_result, training_set.answers[pair_index])
            if loss == 0:
                #print("correct")
                correct_counter += 1
            j_function += loss
        result_tuple: Tuple = (j_function/len(training_set.data), correct_counter/len(training_set.data))
        print("###END OF TEST###")
        return result_tuple
