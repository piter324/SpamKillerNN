from TrainingSet import TrainingSet
from typing import List


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
