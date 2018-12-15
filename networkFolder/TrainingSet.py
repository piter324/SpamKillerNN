from typing import List
import random


class TrainingSet:
    def __init__(self, data: List[List[float]], answers: List[List[float]]):
        assert len(data) == len(answers),\
            "Size of data list(%d) doesn't equal size of correct_answers list(%d)." % (len(data), len(answers))
        self.data: List[List[float]] = data.copy()
        self.answers: List[List[float]] = answers.copy()


# x > y gives true
def generate_random_set1(data_size: int) -> TrainingSet:
    print("###GENERATING RANDOM SET1...###")
    data: List[List[float]] = []
    answers: List[List[float]] = []
    for d in range(data_size):
        data.append([random.random(), random.random()])
        if data[d][0] > data[d][1]:
            answers.append([1])
        else:
            answers.append([-1])
        #print(data[d])
        #print(answers[d])
    print("###END OF GENERATING RANDOM SET1###")
    return TrainingSet(data, answers)


def generate_xor_set(data_size: int) -> TrainingSet:
    print("###GENERATING XOR SET...###")
    data: List[List[float]] = []
    answers: List[List[float]] = []
    for d in range(data_size):
        data.append([random.randint(0, 1), random.randint(0, 1)])
        if (data[d][0] == 1 and data[d][1] == 0) or (data[d][0] == 0 and data[d][1] == 1):
            answers.append([1])
        else:
            answers.append([-1])
        print(data[d])
        print(answers[d])
    print("###END OF GENERATING XOR SET###")
    return TrainingSet(data, answers)


def generate_guess_number() -> TrainingSet:
    print("###GENERATING GUESS THE NUMBER TRIVIAL SET...###")
    data: List[List[float]] = []
    answers: List[List[float]] = []
    random_number = random.random()
    data.append([random_number])
    answers.append([random_number])
    #print(data)
    #print(answers)
    print("###END OF GENERATING GUESS THE NUMBER TRIVIAL SET...###")
    return TrainingSet(data, answers)


