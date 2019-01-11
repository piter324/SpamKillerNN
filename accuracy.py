import csv
import TrainingSet
from NeuralNetwork import NeuralNetwork
from NetworkTester import NetworkTester
if __name__ == '__main__':

    # odpalenie tego pliku przetestuje accuracy dwoch sieci:
    # network1 - siec 30 neuronow w warstwie wejsciowej, 30 w ukrytej, 1 w wyjsciowej (1000 slow)
    # network2 - siec 20 neuronow w warstwie wejsciowej, 20 w ukrytej, 1 w wyjsciowej (500 slow)

    # wczytujemy przetasowane maile
    with open('mails_shuffled.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        mails = list(reader)
    # przygotowanie zbiorow testowych
    data1 = []
    answers1 = []
    data2 = []
    answers2 = []

    # zbior dla network 1
    for m in mails:
        one_data1 = m[1:1001]
        one_answer1 = [m[0]]
        data1.append(one_data1)
        answers1.append(one_answer1)

    test_set1 = TrainingSet.TrainingSet(data1[:1000], answers1[:1000])
    # zbior dla network 2

    for m in mails:
        one_data2 = m[1:501]
        one_answer2 = [m[0]]
        data2.append(one_data2)
        answers2.append(one_answer2)

    test_set2 = TrainingSet.TrainingSet(data2[:1000], answers2[:1000])

    # test network1
    nn1: NeuralNetwork = NeuralNetwork.load("network1")  # wczytanie sieci network1 z pliku
    tester = NetworkTester(nn1)  # utworzeniu obiektu testujacego
    print("Siec network1 ma srednia pewnosc na poziomie: %s%%" % (tester.test_average_certainty(test_set1)*100))
    print("Siec network1 dla progu pewnosci 80%% ma trafnosc na poziomie: %s%%" %
          (tester.test_accuracy(test_set1, 0.8)*100))
    print("Siec network1 dla progu pewnosci 90%% ma trafnosc na poziomie: %s%%" %
          (tester.test_accuracy(test_set1, 0.9) * 100))

    # test network2
    nn2: NeuralNetwork = NeuralNetwork.load("network2")  # wczytanie sieci network2 z pliku
    tester = NetworkTester(nn2)  # utworzenie obiektu testujacego
    print("Siec network2 ma srednia pewnosc na poziomie: %s%%" % (tester.test_average_certainty(test_set2)*100))
    print("Siec network2 dla progu pewnosci 80%% ma trafnosc na poziomie: %s%%" %
          (tester.test_accuracy(test_set2, 0.8)*100))
    print("Siec network2 dla progu pewnosci 90%% ma trafnosc na poziomie: %s%%" %
          (tester.test_accuracy(test_set2, 0.9) * 100))
