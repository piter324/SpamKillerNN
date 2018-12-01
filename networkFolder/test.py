from NeuralNetwork import NeuralNetwork
import Neuron
import Functions
import TrainingSet
from NetworkTester import NetworkTester
import time
import numpy as np
import inspect

#print(inspect.isclass(Functions.FuncAbstract))
#print(inspect.isclass(Functions.Sigmoid))
#print(issubclass(Functions.Sigmoid, Functions.FuncAbstract))

ts: TrainingSet = TrainingSet.generate_random_set1(10)

#v1 = [3, 2, 1]
#v2 = [1, 2, 3]

#print(np.subtract(v1, v2))
#print(np.linalg.norm(np.subtract(v1, v2)))


def f() -> list:
    return [1, 2]


fl: float = f()  # PyCharm sees no problems here :(

#ne = Neuron.Neuron([1, 2, 3, 4], None)
#ne2 = Neuron.Neuron([1, 2, 3, 4], Functions.Sigmoid)
#in1 = [2, -4, 0]

#print(ne.processInput(in1))
#print(ne2.processInput(in1))
#print(in1)

#nn = NeuralNetwork(2, [2, 3], [[1, 2, 3, 4], [-1, -2, -3]],
#                   [None, None])
time.sleep(1)
nn2 = NeuralNetwork(2, [2, 1], [[-1, -2, 3], [3, 2, 1]], [Functions.Sigmoid, None])
time.sleep(1)
nn3 = NeuralNetwork(3, [3, 4, 1],
                    [[1, 3, 2], [4, 6, -2, -3], [3, 2, -1, -2, 3]], [Functions.Sigmoid, Functions.Sigmoid, None])
time.sleep(1)
test_tuple1 = NetworkTester.test(nn2, ts)
print("Test returned:\nTarget function: %s\nCorrectness ratio: %s" % (test_tuple1[0],test_tuple1[1]))
time.sleep(1)
test_tuple2 = NetworkTester.test(nn3, ts)
print("Test returned:\nTarget function: %s\nCorrectness ratio: %s" % (test_tuple2[0], test_tuple2[1]))
#print(nn.process_input([1, 2, 2]))
