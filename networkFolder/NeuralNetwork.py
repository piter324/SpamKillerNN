from typing import List, Callable
import Neuron


class NeuralNetwork:
    def __init__(self, layersAmount: int, neuronsAmount: List[int], weightMatrix: List[List[float]], actFunctions: List[Callable]):
        assert len(neuronsAmount) == layersAmount and len(weightMatrix) == layersAmount and len(actFunctions) == layersAmount #TODO wypisac zmienne
        self.layers: List[List[Neuron.Neuron]] = []
        for layerK in range(layersAmount):
            self.layers.append([])
            for neuronI in range(neuronsAmount[layerK]):
                self.layers[layerK].append(Neuron.Neuron(weightMatrix[layerK], actFunctions[layerK]))
        for k in range(layersAmount):
            print("LAYER %d"%k)
            for i in range(neuronsAmount[k]):
                print(self.layers[k][i].activationFunction)
                print(self.layers[k][i].weights)

    def train(self, trainingSet: list):
        #TODO
        pass

    def processInput(self, inputMatrix: List[float]) -> float:
        result: float = 0
        #TODO do some mathemagics
        return result
