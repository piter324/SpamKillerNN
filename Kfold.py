from NeuralNetwork import NeuralNetwork
from TrainingSet import TrainingSet
from typing import List, Type, Tuple, Union
import Functions
import copy
import pickle


class Kfold:
    def __init__(self, k: int, big_set: TrainingSet, training_name: str):  # set has to be already shuffled
        self.k = k
        self.trainedFolds = 0
        self.nnList: List[Union[NeuralNetwork, None]] = []

        self.results: List[float] = []
        self.stats = []
        self.bigSet = big_set
        self.isInit = False
        self.isStarted = False
        self.trainingName = training_name

    def initBaseNetwork(self, input_size: int, neurons_amount: List[int], weight_interval: Tuple[float, float],
                act_functions: List[Type[Functions.FuncAbstract]], loss_function: Type[Functions.LossFuncAbstract]):
        self.base_network = NeuralNetwork(input_size, neurons_amount, weight_interval, act_functions, loss_function)
        for n in range(self.k):
            self.nnList.append(None)
        self.isInit = True

    def proceedKfold(self, learning_rate: float, iterations_limit: int) -> Union[NeuralNetwork, None]:
        if not self.isInit:
            return None

        self.learningRate = learning_rate
        self.iterationsLimit = iterations_limit
        self.isStarted = True

        while self.trainedFolds < self.k:
            print("### Starting kfold { %d } iteration ###" % self.trainedFolds)
            self.nnList[self.trainedFolds] = copy.deepcopy(self.base_network)
            split_ts = self.bigSet.split(int(self.trainedFolds*(len(self.bigSet.data)/self.k)),
                                         int((self.trainedFolds+1)*(len(self.bigSet.data)/self.k)))
            self.results.append(self.nnList[self.trainedFolds].vTrain(split_ts[0], learning_rate,
                                                                      iterations_limit, split_ts[1]))
            self.trainedFolds += 1
            self.save()
            #input("Press Enter to continue")
        return self.nnList[self.results.index(min(self.results))]

    # TODO
    def continueKfold(self) -> Union[NeuralNetwork, None]:
        if not self.isStarted:
            return None

        return self.proceedKfold(self.learningRate, self.iterationsLimit)

    def save(self):
        file_name = self.trainingName + ".pkl"
        with open(file_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_name: str):
        with open(file_name, 'rb') as input:
            return pickle.load(input)
