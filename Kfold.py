from NeuralNetwork import NeuralNetwork
from TrainingSet import TrainingSet
from typing import List, Type, Tuple, Union
import Functions
import copy
import pickle


class Kfold:
    """
    Class used to perform kfold training

    Attributes
    ----------
    k : int
        kfold k parameter
    trainedFolds : int
        number of folds already trained
    base_network : NeuralNetwork
        base neural network that gets cloned into nnList
    nnList : List[Union[NeuralNetwork, None]]
        list of neural networks that are supposed to be trained
    results : List[float]
        list of validation results for each fully trained network
    stats : List[List[Tuple[float, float]]]
        each element of outer list represents process of training for certain fold
        each element of inner list represents tuple(pair) of target function values
        after training iteration given by index of this list
        first element of tuple is value of target function for training set
        second element of tuple is value of target function for validation set
        example: stats[1][3][0] - value of target function for training set after 3rd iteration for fold 1
    bigset : TrainingSet
        whole training set given to Kfold object, which is going to be split in various ways
    isInit : bool
        flag that says if base_network was set up
    learningRate : Union[float, None]
        learning rate factor for training method
    iterationsLimit : Union[int, None]
        maximal amount of iterations for training method
    isStarted : bool
        flag that says if kfold has been already started
    trainingName : str
        name of this particular kfold object - used for naming file that this object can be saved to

    Methods
    -------
    init_base_network
        Initializes base neural network using given parameters
    proceed_kfold
        Sets up training parameters and performes kfold process
    continue_kfold
        If kfold was already in progress - indicates continuation of kfold process
    save
        saves this object into pickle file
    load
        loads Kfold object from file of given name
    """

    def __init__(self, k: int, big_set: TrainingSet, training_name: str):
        """Initializes fields with initial values."""
        self.k = k
        self.trainedFolds = 0
        self.base_network = None
        self.nnList: List[Union[NeuralNetwork, None]] = []

        self.results: List[float] = []
        self.stats = []
        self.bigSet = big_set
        self.isInit = False
        self.learningRate = None
        self.iterationsLimit = None
        self.isStarted = False
        self.trainingName = training_name

    def init_base_network(self, input_size: int, neurons_amount: List[int], weight_interval: Tuple[float, float],
                          act_functions: List[Type[Functions.FuncAbstract]],
                          loss_function: Type[Functions.LossFuncAbstract]):
        """
        Initializes base neural network. Given parameters are passed forward to NeuralNetwork constructor,
        therefore parameters descriptions are identical as in NeuralNetwork.
        """
        self.base_network = NeuralNetwork(input_size, neurons_amount, weight_interval, act_functions, loss_function)
        for n in range(self.k):
            self.nnList.append(None)
        self.isInit = True

    def proceed_kfold(self, learning_rate: float, iterations_limit: int) -> Union[NeuralNetwork, None]:
        """
        Sets up learning rate and iterations limit and performes kfold process.
        :param learning_rate: learning rate factor for training method
        :param iterations_limit: maximal amount of iterations for training method
        :return: the best neural network - one with lowest validation target function at the end of fold
        """
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
            vtrain_result = self.nnList[self.trainedFolds].vtrain(split_ts[0], learning_rate,
                                                                  iterations_limit, split_ts[1])
            self.results.append(vtrain_result[0])
            self.stats.append(vtrain_result[1])
            self.trainedFolds += 1
            self.save()
        return self.nnList[self.results.index(min(self.results))]

    def continue_kfold(self) -> Union[NeuralNetwork, None]:
        if not self.isStarted:
            return None

        return self.proceed_kfold(self.learningRate, self.iterationsLimit)

    def save(self):
        file_name = self.trainingName + ".pkl"
        with open(file_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_name: str):
        with open(file_name, 'rb') as input:
            return pickle.load(input)
