import sanitizedata as sd
from TrainingSet import TrainingSet
from NeuralNetwork import NeuralNetwork
import Functions

if __name__ == '__main__':
    
    csv_output = sd.prepare_csv("csv01_properUTF8.txt")

    data = []
    answers = []
    for mail in csv_output:
        answers.append([float(mail[0])])
        data_vec = [float(mail[3])] # is formatted or not
        for topicPiece in mail[2]:
            if(len(data_vec) >= 10): break
            data_vec.append(hash(topicPiece)/1000000000000000000) # topic piece per piece            
        for bodyPiece in mail[4]:
            if(len(data_vec) >= 10): break
            data_vec.append(hash(bodyPiece)/1000000000000000000) # message
        while len(data_vec) < 10:
            data_vec.append(0)
        data.append(data_vec)
    
    print(csv_output[3:4])
    print(data[3:4])
    print(answers[3:4])

    training_set = TrainingSet(data, answers)
    neural_network = NeuralNetwork(3, [3, 3, 1], [[1, -2, 3, 2, 1, 2, -1, 2, -2, -2, -1], [2, -1, -3, -1], [-1, -2, 2, 3]], [Functions.Sigmoid, Functions.Sigmoid, Functions.TanH], Functions.DiffSquare)
    #neural_network.make_guess(training_set.data[0])
    neural_network.train(training_set, 0.1, 0.1)