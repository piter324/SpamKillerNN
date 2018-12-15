from NeuralNetwork import NeuralNetwork
import Neuron
import Functions
import TrainingSet
from NetworkTester import NetworkTester
import time
import numpy as np
import inspect
from random import randint as rnd

#a = [[[1,2,3],[3,2,1],[5,6,7]], [[3,2,1],[2,4,5]]]
#b = [[[3,2,1],[2,4,5],[3,3,3]], [[1,2,3],[3,2,1]]]
#print(np.add(a,b))

#print(inspect.isclass(Functions.FuncAbstract))
#print(inspect.isclass(Functions.Sigmoid))
#print(issubclass(Functions.Sigmoid, Functions.FuncAbstract))

#ts: TrainingSet = TrainingSet.generate_random_set1(2)

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
#result_matrix = nn2.examine_single_pair(ts.data[0], ts.answers[0])
#print(result_matrix)

# Training test
#nn2.train(ts, 0.1, 0.8)

#guesser = NeuralNetwork(1, [1], [[0, 0]], [None], Functions.DiffSquare)
#number_set: TrainingSet = TrainingSet.generate_guess_number()
#time.sleep(1)
#guesser.train(number_set, 0.1, 0.8)

# mala siec

#nn4 = NeuralNetwork(2, [2, 1], [[0, 0, 0], [0, 0, 0]], [Functions.Sigmoid, Functions.TanH], Functions.DiffSquare)

#ts5 = TrainingSet.generate_random_set1(20)
#tester = NetworkTester(nn4)
#print(tester.test(ts5))
#time.sleep(2)

#ts4 = TrainingSet.generate_random_set1(200)
#nn4.train(ts4, 0.6, 0.1)

#print(tester.test(ts5))

# XOR

nn8 = NeuralNetwork(3, [2, 2, 1],
                    [[rnd(-3,3), rnd(-3,3), rnd(-3,3)], [rnd(-3,3), rnd(-3,3), rnd(-3,3)], [rnd(-3,3), rnd(-3,3), rnd(-3,3)]],
                    [Functions.Identity, Functions.TanH, Functions.TanH], Functions.DiffSquare)

ts9 = TrainingSet.generate_xor_set(20)
tester3 = NetworkTester(nn8)
print(tester3.test(ts9))
time.sleep(2)

ts8 = TrainingSet.generate_xor_set(200)
nn8.train(ts8, 0.5, 0.1)

print(tester3.test(ts9))

# wieksza siec

#nn6 = NeuralNetwork(3, [2, 2, 1], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
# [Functions.Identity, Functions.Sigmoid, Functions.TanH], Functions.DiffSquare)

#ts7 = TrainingSet.generate_random_set1(20)
#tester2 = NetworkTester(nn6)
#print(tester2.test(ts7))
#time.sleep(3)

#ts6 = TrainingSet.generate_random_set1(200)
#nn6.train(ts6, 1, 0.1)

#print(tester2.test(ts7))