from typing import List
import random


class TrainingSet:
    def __init__(self, data: List[List[float]], answers: List[List[float]]):
        assert len(data) == len(answers),\
            "Size of data list(%d) doesn't equal size of correct_answers list(%d)." % (len(data), len(answers))
        self.data: List[List[float]] = data.copy()
        self.answers: List[List[float]] = answers.copy()


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
        print(data[d])
        print(answers[d])
    print("###END OF GENERATING RANDOM SET1###")
    return TrainingSet(data, answers)
