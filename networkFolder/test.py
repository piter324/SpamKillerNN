import Neuron
from NeuralNetwork import NeuralNetwork
import time

ne = Neuron.Neuron([1, 2, 3, 4], None)
ne2 = Neuron.Neuron([1, 2, 3, 4], Neuron.sigmoid)
in1 = [2, -4, 0]

#print(ne.processInput(in1))
#print(ne2.processInput(in1))
#print(in1)

nn = NeuralNetwork(2, [2, 3], [[1, 2, 3, 4], [-1, -2, -3]],
                   [None, None])
time.sleep(1)
print(nn.process_input([1, 2, 2]))
