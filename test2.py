from NeuralNetwork import NeuralNetwork
from NetworkTester import NetworkTester

# przed wszytkim trzeba przygotowac testing_set = ... to ma byc ten set z 1000 pierwszych maili,
# prawdopodobnie w kodzie do kfolda (w examination.py) masz cos podobnego jak to:
# test_set = TrainingSet.TrainingSet(data[:1000], answers[:1000])
# to to wlasnie o to chodzi

# no i teraz chcemy przetestowac oddzielnie kazda siec poniewaz bedziesz musial za kazdym razem zmieniac liczbe slow(!)
# na odpowiednia tam we wczytywaniu maili z csv

# siec 1 - 2warstwowa
nn1 = NeuralNetwork.load("")  # wpisz w "" nazwe pliku sieci 2warstwowej
tester1 = NetworkTester(nn1)
print(tester1.test(testing_set))

# siec 2 - 3warstwowa 20:20:1
# nn2 = NeuralNetwork.load("")  # wpisz w "" nazwe pliku sieci 3warst 20:20:1
# tester2 = NetworkTester(nn2)
# print(tester2.test(testing_set))

# siec 3 - 3warstwowa 30:30:1
# nn3 = NeuralNetwork.load("")  # wpisz w "" nazwe pliku sieci 3warst 30:30:1
# tester3 = NetworkTester(nn3)
# print(tester3.test(testing_set))