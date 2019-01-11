from NeuralNetwork import NeuralNetwork
import Functions
import TrainingSet
from NetworkTester import NetworkTester

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

    ts = TrainingSet.generate_rgb3(200)

    nn = NeuralNetwork(3, [5, 3, 3], (-0.05, 0.05), [Functions.Identity, Functions.Sigmoid, Functions.Sigmoid],
                       Functions.DiffSquare)

    nn.train(ts, 0.2, 0.01, 3000)

    ts1 = TrainingSet.generate_rgb3(2000)

    tester = NetworkTester(nn)
    print(tester.test(ts1))
    print(nn.make_guess([1, 28, -14]))
    print(nn.make_guess([27, 1, -15]))
    print(nn.make_guess([-10, 0, 18]))

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
