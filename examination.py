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
    # with open('mails_with_word_occurences.csv') as file:
    #     reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    #     mails = list(reader)
    #
    # shuffle(mails)
    #
    # # for m in range(len(mails)-10, len(mails)-1, 1):
    # #     print("Mail %d" % m)
    # #     print(mails[m])
    # #     print(len(mails[m]))
    # # /wczytanie maili
    # # przygotowanie training set
    # data = []
    # answers = []
    # for m in mails:
    #     one_data = m[1:501]
    #     # print(len(one_data))
    #     one_answer = [m[0]]
    #     # print(type(one_answer[0]))
    #     data.append(one_data)
    #     answers.append(one_answer)
    #
    # # print(answers)
    # # print(data[0])
    # # print(data[1])
    # mails_set = TrainingSet.TrainingSet(data[2000:3000], answers[2000:3000])
    # test_set = TrainingSet.TrainingSet(data[:1000], answers[:1000])

    # /przygotowanie training set
    # stworzenie wytrenowanie i test

    # kfold = Kfold(5, mails_set, "kfold1")
    # kfold.init_base_network(500, [20, 20, 1], (-0.1, 0.1), [Functions.Identity, Functions.Sigmoid, Functions.Sigmoid], Functions.DiffSquare)
    #
    # kfold_result = kfold.proceed_kfold(2, 200)

    # network_tester = NetworkTester(kfold_result)
    # print(network_tester.test(test_set)[0])
    #/stworzenie wytrenowanie i test

    # ----zaladowanie kfolda i kontynuowanie trenowania----
    # kfold: Kfold = Kfold.load("kfold1.pkl")
    # kfold_result: NeuralNetwork = kfold.continue_kfold()
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


    # ----statystyki ----
    # stats = kfold.stats
    #
    # dane_wykresu0 = stats[0]
    # dane_wykresu1 = stats[1]
    #
    # t = dane_wykresu0[0][0]  # wartosc pierwszej liczby z pary dla x = 0 (x - iteracja)
    # v = dane_wykresu0[0][1]

    # sprawdzenie
    # print(kfold.nnList[0].make_guess(test_set.data[0]))
    # print(test_set.answers[0])
    # print(kfold.nnList[0].make_guess(test_set.data[1]))
    # print(test_set.answers[1])
    # print(kfold.nnList[0].make_guess(test_set.data[2]))
    # print(test_set.answers[2])
    #
    # tester = NetworkTester(kfold.nnList[0])
    # print(tester.test(test_set))

    # kfold.nnList[0].save("spam1.pkl")

    # spam1 = NeuralNetwork.load("spam1.pkl")
    #
    # print(spam1.make_guess(test_set.data[0]))
    # print(test_set.answers[0])

    # SPRAWDZENIE CZY DALEJ DZIALA
    # XOR

    # ts = TrainingSet.generate_xor_set(150)
    #
    # nn = NeuralNetwork(2, [4, 2, 1], (-1, 1), [Functions.Identity, Functions.TanH, Functions.TanH], Functions.DiffSquare)
    #
    # nn.train(ts, 0.2, 0.001)
    #
    # ts1 = TrainingSet.generate_xor_set(2000)
    #
    # tester = NetworkTester(nn)
    # print(tester.test(ts1))
    # print(nn.make_guess([1, 1]))
    # print(nn.make_guess([0, 1]))
    # print(nn.make_guess([1, 0]))
    # print(nn.make_guess([0, 0]))

    # RGB

    ts = TrainingSet.generate_rgb3(50)

    nn = NeuralNetwork(3, [3, 3], (-0.05, 0.05), [Functions.Sigmoid, Functions.Sigmoid], Functions.DiffSquare)

    nn.train(ts, 0.1, 0.001, 15000)

    ts1 = TrainingSet.generate_rgb3(2000)

    tester = NetworkTester(nn)
    print(tester.test(ts1))
    print(nn.make_guess([1, 28, -14]))
    print(nn.make_guess([27, 1, -15]))
    print(nn.make_guess([-10, 0, 18]))

