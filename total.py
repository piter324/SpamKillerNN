import sanitizedata as sd
import TrainingSet
from NeuralNetwork import NeuralNetwork
from NetworkTester import NetworkTester
import Functions
from random import random
import time
import csv
from Kfold import Kfold

if __name__ == '__main__':
    
    # csv_output = sd.prepare_csv("csv01_properUTF8.txt")
    #
    # data = []
    # answers = []
    # for mail in csv_output:
    #     answers.append([float(mail[0])])
    #     data_vec = [] # is formatted or not
    #     # data_vec = [float(mail[3])] # is formatted or not
    #     for topicPiece in mail[2]:
    #         if(len(data_vec) >= 30): break
    #         data_vec.append(hash(topicPiece)/1000000000000000000) # topic piece per piece
    #     for bodyPiece in mail[4]:
    #         if(len(data_vec) >= 30): break
    #         data_vec.append(hash(bodyPiece)/1000000000000000000) # message
    #     while len(data_vec) < 30:
    #         data_vec.append(0)
    #     data.append(data_vec)
    
    #print(csv_output[3:4])
    #print(data[3:4])
    #print(answers[3:4])

    # wczytanie maili
    # with open('mails_with_word_occurences.csv') as file:
    #     reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    #     mails = list(reader)
    #
    # for m in range(len(mails)-10,len(mails)-1,1):
    #     print("Mail %d" % m)
    #     print(mails[m])
    #     print(len(mails[m]))
    # # # /wczytanie maili
    # # # przygotowanie training set
    # data = []
    # answers = []
    # for m in mails:
    #     one_data = m[1:501]
    #     print(len(one_data))
    #     one_answer = [m[0]]
    #     print(type(one_answer[0]))
    #     data.append(one_data)
    #     answers.append(one_answer)
    #
    # print(answers)
    # print(data[0])
    # print(data[1])
    # mails_set = TrainingSet.TrainingSet(data[2000:3000], answers[2000:3000])

    # przygotowanie training set
    # stworzenie wytrenowanie i test
    # w0 = []
    # for i in range(501):
    #     w0.append(random()/50 - 0.01)
    #
    # w1 = []
    # for i in range(51):
    #     w1.append(random()/50 - 0.01)
    #
    # w2 = []
    # for i in range(21):
    #     w2.append(random()/50 - 0.01)
    #
    # neural_network = NeuralNetwork(500, [50, 1], (-0.01, 0.01), [Functions.Sigmoid, Functions.Sigmoid],
    #                                Functions.DiffSquare)
    #
    # # nn2 = NeuralNetwork(6, [3,4,2], (-0.1, 0.1), [Functions.Sigmoid, Functions.Sigmoid, Functions.Sigmoid], Functions.DiffSquare)
    # neural_network.train(mails_set,1,0.005)
    #
    # print(mails[4])
    # print(neural_network.make_guess(data[4]))
    # print(mails[3600])
    # print(neural_network.make_guess(data[3600]))
    # /stworzenie wytrenowanie i test

    # weight_vector = []
    # for i in range(31):
    #     weight_vector.append(random())
    #
    # print(weight_vector)
    #
    # weight_vector2 = []
    # for i in range(51):
    #     weight_vector2.append(random())
    # print(weight_vector2)
    # time.sleep(2)

    # weight_vector3 = []
    # for i in range(3):
    #     weight_vector3.append(random()*2 - 1)
    # print(weight_vector3)
    #
    # weight_vector4 = []
    # for i in range(4):
    #     weight_vector4.append(random()*2 - 1)
    # print(weight_vector4)
    #
    # w5 = []
    # for i in range(5):
    #     w5.append(random()*2 - 1)
    # print(w5)

    # training_set = TrainingSet.TrainingSet(data[6:600], answers[6:600])
    # print(len(data))
    # print(len(answers))
    # time.sleep(2)
    # neural_network = NeuralNetwork(3, [50, 3, 1], [weight_vector, weight_vector2, [0.5, -0.5, -1, 1]],
    #                                [Functions.Identity, Functions.Identity, Functions.TanH], Functions.DiffSquare)
    # #neural_network.make_guess(training_set.data[0])
    # neural_network.train(training_set, 0.1, 0.2)

    # for t in range(6):
    #     print(neural_network.make_guess(training_set.data[t]))
    #     print(answers[t])
    #     print("--------")

    # XOR test again
    # nn = NeuralNetwork(4, [3, 3, 4, 1], [weight_vector3, weight_vector4, [1,-2,2,1], w5],
    #                    [Functions.Identity, Functions.Identity, Functions.Identity, Functions.TanH], Functions.DiffSquare)
    #
    # w1 = []
    # for w in range(3):
    #     w1.append(random()*4 - 2)
    #
    # w2 = []
    # for w in range(3):
    #     w2.append(random() * 2 - 1)
    #
    # w3 = []
    # for w in range(3):
    #     w3.append(random() * 2 - 1)
    #
    # nn2 = NeuralNetwork(3, [2,2,1], [w1, w2, w3],[Functions.Identity, Functions.TanH, Functions.TanH],
    # Functions.DiffSquare)
    #
    # ts = TrainingSet.generate_xor_set(500)
    # nn2.train(ts, 0.5, 0.0001)
    # print(nn2.make_guess([1, 1]))
    # print(nn2.make_guess([0, 1]))
    # print(nn2.make_guess([1, 0]))
    # print(nn2.make_guess([0, 0]))

    # RGB test

    # w1 = []
    # for w in range(4):
    #     w1.append(random()-0.5)
    #
    # w2 = []
    # for w in range(4):
    #     w2.append(random()-0.5)
    #
    # w3 = []
    # for w in range(8):
    #     w3.append(random()-0.5)
    # nn3 = NeuralNetwork(2, [3, 3], [w1,[-1,0,1,0.5]], [Functions.Sigmoid,
    #                                                                Functions.Sigmoid], Functions.DiffSquare)

    # ts2 = TrainingSet.generate_rgb3(8)
    #
    # print(ts2.data)
    # print(ts2.answers)
    #
    # splitTS = ts2.split(0, 2)
    # print("1")
    # print(splitTS[0].data)
    # print("3")
    # print(splitTS[0].answers)
    # print("2")
    # print(splitTS[1].data)
    # print("4")
    # print(splitTS[1].answers)

    # nn3.train(ts2, 0.2, 0.04)
    #
    # print(nn3.make_guess([25,5,-12]))
    # print(nn3.make_guess([1,-16,23]))
    # print(nn3.make_guess([2,24,18]))

    #guess the number

    # nn4 = NeuralNetwork(3, [1,3,1], [[1,2], [-1,1], [-2,2,3,-1]], [Functions.Identity, Functions.Identity, Functions.Identity],
    #                     Functions.DiffSquare)
    #
    # ts3 = TrainingSet.generate_guess_number()
    #
    # nn4.train(ts3,0.0001,0.00005)
    #
    # print(ts3.data[0])
    # print(nn4.make_guess(ts3.data[0]))

    # nn5 = NeuralNetwork(3, [3, 3, 1], [[0.5,0.2,-0.3,0.45], [0.3,-0.5,-0.4,-0.2],[0.4,0.2,0.7,-0.8]], [Functions.Identity,
    #                     Functions.Identity,Functions.Identity], Functions.DiffSquare)
    #
    # ts4 = TrainingSet.generate_sum_set(200)
    #
    # nn5.train(ts4,0.1,0.000002)
    #
    # print(nn5.make_guess([0.2,0.3,-0.4]))

    # KFOLD TEST
    # RGB

    ts = TrainingSet.generate_rgb3(200)

    kfold_object = Kfold(5, ts)
    kfold_object.initNetworks(3, [3, 3], (-2, 2), [Functions.Sigmoid, Functions.Sigmoid], Functions.DiffSquare)

    best_network = kfold_object.proceedKfold(0.5, 500)

    print(best_network.make_guess([10,25,-4]))
