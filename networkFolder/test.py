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

ts: TrainingSet = TrainingSet.generate_random_set1(1)

#ne = Neuron.Neuron([1, 2, 3, 4], None)
#ne2 = Neuron.Neuron([1, 2, 3, 4], Functions.Sigmoid)
#in1 = [2, -4, 0]

#print(ne.processInput(in1))
#print(ne2.processInput(in1))
#print(in1)

#nn = NeuralNetwork(2, [2, 3], [[1, 2, 3, 4], [-1, -2, -3]],
#                   [None, None])
time.sleep(1)
nn2 = NeuralNetwork(2, [2, 1], [[-1, -2, 3], [3, 2, 1]], [Functions.Sigmoid, None], Functions.DiffSquare)
#time.sleep(1)
#nn3 = NeuralNetwork(3, [3, 4, 1], [[1, 3, 2], [4, 6, -2, -3], [3, 2, -1, -2, 3]],
                    #[Functions.Sigmoid, Functions.Sigmoid, None], Functions.DiffSquare)
time.sleep(1)

#nt2 = NetworkTester(nn2)
#nt3 = NetworkTester(nn3)

#test_tuple1 = nt2.test(ts)
#print("Test returned:\nTarget function: %s\nCorrectness ratio: %s" % (test_tuple1[0],test_tuple1[1]))
#time.sleep(1)
#test_tuple2 = nt3.test(ts)
#print("Test returned:\nTarget function: %s\nCorrectness ratio: %s" % (test_tuple2[0], test_tuple2[1]))
#print(nn.process_input([1, 2, 2]))

# BackProp test
result_matrix = nn2.examine_single_pair(ts.data[0], ts.answers[0])
print(result_matrix)
