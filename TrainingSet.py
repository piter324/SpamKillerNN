from typing import List
import random


class TrainingSet:
    """
    Representation of training pairs as list of data and list of correct answers

    Attributes
    ----------
    data : List[List[float]]
        list of input data (represented as vectors)
    answers : List[List[float]]
        list of answers (represented as vectors)

    Methods
    -------
    split
        splits this training set into two sets
    """

    def __init__(self, data: List[List[float]], answers: List[List[float]]):
        assert len(data) == len(answers),\
            "Size of data list(%d) doesn't equal size of correct_answers list(%d)." % (len(data), len(answers))
        self.data: List[List[float]] = data.copy()
        self.answers: List[List[float]] = answers.copy()

    def split(self, start: int, end: int):
        """
        Splits this training set into two new sets and return them. This method doesn't change the original set.
        :param start: Index of first 'cut point'
        :param end: Index of second 'cut point'
        :return: Two training sets - first is from index 0 to start concatenated to end to last index,
            second is from start to end
        """

        data1 = self.data[:start] + self.data[end:]
        answers1 = self.answers[:start] + self.answers[end:]

        data2 = self.data[start:end]
        answers2 = self.answers[start:end]

        return [TrainingSet(data1, answers1), TrainingSet(data2, answers2)]


# functions below were used only to test neural network during development, they generate sets of simple problems

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
    print("###END OF GENERATING XOR SET###")
    return TrainingSet(data, answers)


def generate_guess_number() -> TrainingSet:
    print("###GENERATING GUESS THE NUMBER TRIVIAL SET...###")
    data: List[List[float]] = []
    answers: List[List[float]] = []
    random_number = random.random()
    data.append([random_number])
    answers.append([random_number])
    print("###END OF GENERATING GUESS THE NUMBER TRIVIAL SET###")
    return TrainingSet(data, answers)


def generate_rgb3(data_size: int) -> TrainingSet:
    print("###GENERATING RGB3 SET...###")
    data: List[List[float]] = []
    answers: List[List[float]] = []
    for c in range(data_size):
        color = []
        for s in range(3):
            color.append(random.randint(-30, 30))
        data.append(color)
        if color[0] > color[1] and color[0] > color[2]:
            answers.append([1, 0, 0])
        elif color[1] > color[0] and color[1] > color[2]:
            answers.append([0, 1, 0])
        else:
            answers.append([0, 0, 1])
    print("###END OF GENERATING RGB3 SET")
    return TrainingSet(data, answers)


def generate_sum_set(data_size: int):
    print("###GENERATING SUM SET..###")
    data = []
    answers = []
    for d in range(data_size):
        triad = []
        for n in range(3):
            triad.append(random.random()-0.5)
        data.append(triad)
        s = 0
        for n in triad:
            s += n
        answers.append([s])
    print("###END OF GENERATING SUM SET###")
    return TrainingSet(data, answers)
