from TrainingSet import TrainingSet
from typing import List
import math


class NetworkTester:
    def __init__(self, neural_network):
        self.tested_network = neural_network

    def test(self, testing_set: TrainingSet) -> float:
        """
        Performes test of tested_network - calculates value of target function for given training set,
        which is defined as average value of loss_functions for each training pair.
        :param testing_set: list of training pairs, for which this method calculates target function value
        :return: value of target function
        """

        # Checking if training_set is legal for neural_network
        assert len(testing_set.data[0]) == len(self.tested_network.layers[0][0].weights)-1
        assert len(testing_set.answers[0]) == len(self.tested_network.layers[len(self.tested_network.layers)-1])

        # Target function (that's ought to be minimized) J = (1/N) * (Σ t=1 to N (loss_function))
        # where t is index of one training pair and N is amount of all training pairs
        j_sum: float = 0
        for pair_index in range(len(testing_set.data)):  # for every single training pair
            # take guess from tested_network for current data from testing_set
            single_result: List[float] = self.tested_network.make_guess(testing_set.data[pair_index])
            # calculate loss (for received guess) with network's loss formula and add result to j_sum
            j_sum += self.tested_network.loss_function.func(single_result, testing_set.answers[pair_index])
        return j_sum/len(testing_set.data)  # devide j_sum by amount of training pairs and return the result

    def test_average_certainty(self, testing_set: TrainingSet):
        """
        Average certainty test, assuming answer is a vector with one value: 0 or 1. For this case it is square root
        of average value of loss function for given testing set.
        :param testing_set: list of pairs for which this method calculates average certainty
        :return: percentage value of average certainty
        """

        return 1-math.sqrt(self.test(testing_set))

    def test_accuracy(self, testing_set: TrainingSet, threshold_percent: float):
        """
        Accuracy test, assuming answer is a vector with one value: 0 or 1.
        :param testing_set: list of pairs for which this method calculates accuracy
        :param threshold_percent: percent threshold value, for which we consider network as sure of its answer
            example: threshold_percent == 0.8 and the correct answer is [0] - if network answered 0.3 method considers
            such answer as wrong even though it's closer to 0 than to 1. If network answered 0.19 method considers
            such answer as correct.
        :return: accuracy of tested neural network for given testing set
        """

        # Checking if training_set is legal for neural_network
        assert len(testing_set.data[0]) == len(self.tested_network.layers[0][0].weights) - 1
        assert len(testing_set.answers[0]) == 1

        correct_counter: int = 0
        for pair_index in range(len(testing_set.data)):
            # take guess from tested_network for current data from testing_set
            single_result: List[float] = self.tested_network.make_guess(testing_set.data[pair_index])
            # check if answer is correct
            if abs(single_result[0] - testing_set.answers[pair_index][0]) <= 1-threshold_percent:
                correct_counter += 1

        return correct_counter/len(testing_set.data)
