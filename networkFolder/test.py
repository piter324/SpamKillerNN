#from Neuron import Neuron
import Neuron
from NeuralNetwork import NeuralNetwork
import numpy as np

class number:
    m = None
    def val(self, v):
        self.m = v
    def p(self):
        print(self.m)


n: int = 1

def sum(a,b):
    return a+b

def mul(a,b):
    return a*b

def doSomething(function,a,b):
    return function(a,b)

#print(doSomething(sum,1,2))

#print(doSomething(mul,1,2))

ne=Neuron.Neuron([1, 2, 3, 4], None)
ne2=Neuron.Neuron([1, 2, 3, 4], Neuron.sigmoid)
in1=[2,-4,0]
#help(Neuron)
#help(type(self))



#c=1
#a = [1,2,3,4]
#b = [2,4,0,1]
#print(np.dot(a,b))
#print(type(c) is int)
#if type(a) is list:
#    print("hello")

#print(ne.test(3))
#print(ne2.test(4))

#print(ne.processInput(in1))
#print(ne2.processInput(in1))
#print(in1)

nn = NeuralNetwork(2, [3, 3], [[1, 2, 3], [-1, -2, -3]], [Neuron.sigmoid, None])
#nn2 = NeuralNetwork(3, [3, 3], [3, 4, 5], [Neuron.sigmoid, Neuron.identity])