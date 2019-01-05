import csv
from random import shuffle
import TrainingSet
from NeuralNetwork import NeuralNetwork
import Functions
from Kfold import Kfold
from NetworkTester import NetworkTester

if __name__ == '__main__':

    # MAILS SHUFFLED

    #wczytanie maili
    with open('mails_with_word_occurences.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        mails = list(reader)

    shuffle(mails)

    # for m in range(len(mails)-10, len(mails)-1, 1):
    #     print("Mail %d" % m)
    #     print(mails[m])
    #     print(len(mails[m]))
    # /wczytanie maili
    # przygotowanie training set
    data = []
    answers = []
    for m in mails:
        one_data = m[1:501]
        # print(len(one_data))
        one_answer = [m[0]]
        # print(type(one_answer[0]))
        data.append(one_data)
        answers.append(one_answer)

    # print(answers)
    # print(data[0])
    # print(data[1])
    mails_set = TrainingSet.TrainingSet(data[2000:3000], answers[2000:3000])
    test_set = TrainingSet.TrainingSet(data[1000:2000], answers[1000:2000])

    # /przygotowanie training set
    # stworzenie wytrenowanie i test

    kfold = Kfold(5, mails_set, "kfold1")
    kfold.initBaseNetwork(500, [20, 1], (-0.01, 0.01), [Functions.Sigmoid, Functions.Sigmoid], Functions.DiffSquare)

    kfold_result = kfold.proceedKfold(1, 100)

    network_tester = NetworkTester(kfold_result)
    print(network_tester.test(test_set)[0])
    #/stworzenie wytrenowanie i test

    # ----zaladowanie kfolda i kontynuowanie trenowania----
    # kfold: Kfold = Kfold.load("kfold1.pkl")
    # kfold_result: NeuralNetwork = kfold.continueKfold()
    #
    # kfold_result.save("network")

    # ----pobranie statystyk----
    # stats = kfold.stats
    #
    # print(stats)


    # eksperymenty
    # network_tester = NetworkTester(kfold.nnList[0])
    # print(network_tester.test(test_set)[0])

    # print(kfold.nnList[0].make_guess(test_set.data[0]))
    # print(test_set.answers[0])

    # network = kfold.nnList[0]
    #
    # network.train((mails_set.split(200, 500))[1], 1, 0.04)
    #
    # network.save("net1")

    # network = NeuralNetwork.load("net1")
    #
    # tester = NetworkTester(network)
    # print(tester.test(test_set))
