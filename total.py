import sanitizedata as sd
from TrainingSet import TrainingSet
from NeuralNetwork import NeuralNetwork
from NetworkTester import NetworkTester
import Functions
from random import random
import time

if __name__ == '__main__':
    
    csv_output = sd.prepare_csv("csv01_properUTF8.txt")

    data = []
    answers = []
    for mail in csv_output:
        answers.append([float(mail[0])])
        data_vec = [] # is formatted or not
        # data_vec = [float(mail[3])] # is formatted or not
        for topicPiece in mail[2]:
            if(len(data_vec) >= 30): break
            data_vec.append(hash(topicPiece)/1000000000000000000) # topic piece per piece            
        for bodyPiece in mail[4]:
            if(len(data_vec) >= 30): break
            data_vec.append(hash(bodyPiece)/1000000000000000000) # message
        while len(data_vec) < 30:
            data_vec.append(0)
        data.append(data_vec)
    
    #print(csv_output[3:4])
    #print(data[3:4])
    #print(answers[3:4])

    weight_vector = []
    for i in range(31):
        weight_vector.append(random())

    print(weight_vector)

    weight_vector2 = []
    for i in range(51):
        weight_vector2.append(random())
    print(weight_vector2)
    time.sleep(2)

    training_set = TrainingSet(data[6:600], answers[6:600])
    print(len(data))
    print(len(answers))
    time.sleep(2)
    neural_network = NeuralNetwork(3, [50, 3, 1], [weight_vector, weight_vector2, [0.5, -0.5, -1, 1]],
                                   [Functions.Identity, Functions.Identity, Functions.TanH], Functions.DiffSquare)
    #neural_network.make_guess(training_set.data[0])
    neural_network.train(training_set, 0.1, 0.2)

    for t in range(6):
        print(neural_network.make_guess(training_set.data[t]))
        print(answers[t])
        print("--------")