import Functions
from TrainingSet import TrainingSet
from typing import List, Tuple, Type
import numpy as np


class NetworkTester:
    def __init__(self, neural_network):
        self.tested_network = neural_network

    def calc_loss(self, guess: List[float], answer: List[float]) -> float:
        """Calculates loss with network's loss formula"""
        print("Guess: %s" % guess)
        print("Answer: %s" % answer)
        return self.tested_network.loss_function.func(guess, answer)

    def test(self, training_set: TrainingSet) -> Tuple[float, float]:
        # Checking if training_set is legal for neural_network
        assert len(training_set.data[0]) == len(self.tested_network.layers[0][0].weights)-1
        assert len(training_set.answers[0]) == len(self.tested_network.layers[len(self.tested_network.layers)-1])
        print("###STARTING TEST...###")
        # Target function (to minimize) J = (1/N) * Σ t=1 to N ||answer_t - guess_t||^2
        j_sum: float = 0
        correct_counter: int = 0
        for pair_index in range(len(training_set.data)):
            single_result: List[float] = self.tested_network.make_guess(training_set.data[pair_index])
            loss: float = NetworkTester.calc_loss(self, single_result, training_set.answers[pair_index])
            if loss == 0:
                #print("correct")
                correct_counter += 1
            j_sum += loss
        result_tuple: Tuple = (j_sum/len(training_set.data), correct_counter/len(training_set.data))
        print("###END OF TEST###")
        return result_tuple
