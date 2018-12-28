from NeuralNetwork import NeuralNetwork
from TrainingSet import TrainingSet
from typing import List, Type, Tuple, Union
import Functions


class Kfold:
    def __init__(self, k: int, bigSet: TrainingSet): # set has to be already shuffled
        self.k = k
        self.nnList: List[NeuralNetwork] = []
        self.bigSet = bigSet
        self.isInit = False

    # TODO make deep copies
    def initNetworks(self, input_size: int, neurons_amount: List[int], weight_interval: Tuple[float, float],
                act_functions: List[Type[Functions.FuncAbstract]], loss_function: Type[Functions.LossFuncAbstract]):
        for n in range(self.k):
            self.nnList.append(NeuralNetwork(input_size, neurons_amount, weight_interval, act_functions, loss_function))
        self.isInit = True

    def proceedKfold(self, learning_rate: float, iterations_limit: int) -> Union[NeuralNetwork, None]:
        if not self.isInit:
            return None
        results: List[float] = []
        for n in range(self.k):
            print("### Starting kfold { %d } iteration ###" % n)
            split_ts = self.bigSet.split(int(n*(len(self.bigSet.data)/self.k)), int((n+1)*(len(self.bigSet.data)/self.k)))
            results.append(self.nnList[n].vTrain(split_ts[0], learning_rate, iterations_limit, split_ts[1]))
            #input("Press Enter to continue")
        return self.nnList[results.index(max(results))]
