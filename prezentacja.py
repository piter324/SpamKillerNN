from NeuralNetwork import NeuralNetwork
import Functions
import TrainingSet
from NetworkTester import NetworkTester
import csv

if __name__ == '__main__':

    # XOR

    # ts = TrainingSet.generate_xor_set(150)
    #
    # nn = NeuralNetwork(2, [2, 2, 1], (-1, 1), [Functions.Identity, Functions.TanH, Functions.TanH], Functions.DiffSquare)
    #
    # nn.train(ts, 0.2, 0.001, 2000)
    #
    # ts1 = TrainingSet.generate_xor_set(2000)
    #
    # tester = NetworkTester(nn)
    # print(tester.test(ts1))
    # print(nn.make_guess([1, 1]))
    # print(nn.make_guess([0, 1]))
    # print(nn.make_guess([1, 0]))
    # print(nn.make_guess([0, 0]))

    # XOR 2

    # ts = TrainingSet.generate_xor_set(150)
    #
    # nn = NeuralNetwork(2, [4, 1], (-1, 1), [Functions.TanH, Functions.TanH],
    #                    Functions.DiffSquare)
    #
    # nn.train(ts, 0.2, 0.001, 2000)
    #
    # ts1 = TrainingSet.generate_xor_set(2000)
    #
    # tester = NetworkTester(nn)
    # print(tester.test(ts1))
    # print(nn.make_guess([1, 1]))
    # print(nn.make_guess([0, 1]))
    # print(nn.make_guess([1, 0]))
    # print(nn.make_guess([0, 0]))

    # x > y

    # ts = TrainingSet.generate_random_set1(200)
    # nn = NeuralNetwork(2, [2, 1], (-1, 1), [Functions.TanH, Functions.TanH], Functions.DiffSquare)
    # nn.train(ts, 0.2, 0.001, 1000)
    #
    # ts1 = TrainingSet.generate_random_set1(1000)
    # tester = NetworkTester(nn)
    # print(tester.test(ts1))
    # print(nn.make_guess([0.5, 1]))

    # RGB

    # ts = TrainingSet.generate_rgb3(200)
    #
    # nn = NeuralNetwork(3, [3, 3], (-0.05, 0.05), [Functions.Sigmoid, Functions.Sigmoid], Functions.DiffSquare)
    #
    # nn.train(ts, 0.4, 0.01, 3000)
    #
    # ts1 = TrainingSet.generate_rgb3(2000)
    #
    # tester = NetworkTester(nn)
    # print(tester.test(ts1))
    # print(nn.make_guess([1, 28, -14]))
    # print(nn.make_guess([27, 1, -15]))
    # print(nn.make_guess([-10, 0, 18]))

    # RGB 2

    # ts = TrainingSet.generate_rgb3(200)
    #
    # nn = NeuralNetwork(3, [5, 3, 3], (-0.05, 0.05), [Functions.Identity, Functions.Sigmoid, Functions.Sigmoid],
    #                    Functions.DiffSquare)
    #
    # nn.train(ts, 0.2, 0.01, 3000)
    #
    # ts1 = TrainingSet.generate_rgb3(2000)
    #
    # tester = NetworkTester(nn)
    # print(tester.test(ts1))
    # print(nn.make_guess([1, 28, -14]))
    # print(nn.make_guess([27, 1, -15]))
    # print(nn.make_guess([-10, 0, 18]))

    # sum

    # ts = TrainingSet.generate_sum_set(500)
    # nn = NeuralNetwork(3, [1, 1], (-1, 1), [Functions.Identity, Functions.Identity],
    #                    Functions.DiffSquare)
    # nn.train(ts, 0.5, 0.000001, 1000)
    #
    # ts1 = TrainingSet.generate_sum_set(1000)
    # tester = NetworkTester(nn)
    # print(tester.test(ts1))
    # print(nn.make_guess([0.25, 0.6, -0.15]))

    # maile

    with open('mails_shuffled.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        mails = list(reader)
    # /wczytanie maili
    # przygotowanie training set
    data = []
    answers = []
    for m in mails:
        one_data = m[1:501]
        one_answer = [m[0]]
        data.append(one_data)
        answers.append(one_answer)

    mails_set = TrainingSet.TrainingSet(data[2000:3000], answers[2000:3000])
    test_set = TrainingSet.TrainingSet(data[:1000], answers[:1000])
    # \przygotowanie training set

    # nn = NeuralNetwork(200, [20, 1], (-0.01, 0.01), [Functions.Sigmoid, Functions.Sigmoid], Functions.DiffSquare)
    # nn.train(mails_set, 1, 0.01, 200)
    # nn.save("example1")

    # nn: NeuralNetwork = NeuralNetwork.load("example1")
    # print(nn.make_guess(test_set[0]))
    # print(nn.make_guess(test_set[1]))
    # print(nn.make_guess(test_set[3250]))
    # print(nn.make_guess(test_set[3500]))

    # nn1: NeuralNetwork = NeuralNetwork.load("network1")
    # tester = NetworkTester(nn1)
    # print(tester.test(test_set))
    # print(tester.test_average_certainty(test_set))
    # print(tester.test_accuracy(test_set, 0.97))

    nn2: NeuralNetwork = NeuralNetwork.load("network2")
    tester = NetworkTester(nn2)
    print(tester.test(test_set))
    print(tester.test_average_certainty(test_set))
    print(tester.test_accuracy(test_set, 0.9))
