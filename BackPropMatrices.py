from typing import List, Union


class BackPropMatrices:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.dq_dykj_matrix: List[List[Union[float, None]]] = []  # dq/dykj
        self.afunc_derivs_matrix: List[List[float]] = []  # ∂act_funckj(skj)/∂skj skj - sum from k-th layer's j-th neuron
        self.y: List[List[float]] = []  # y[0] - input, y[k] k=1,2,... - output of k-1-th layer

        # initialize dq_dykj_matrix with Nones for each layer except last one
        # (which has to be calculated and appended with init_last_dq_dykj() method)
        for layerK in range(len(neural_network.layers)-1):
            dq_dykj_vector: List[None] = []
            for neuronJ in range(len(neural_network.layers[layerK])):
                dq_dykj_vector.append(None)
            self.dq_dykj_matrix.append(dq_dykj_vector)

        # initialize afunc_derivs_matrix with Nones
        for layerK in range(len(neural_network.layers)):
            afunc_derivs_vector: List[None] = []
            for neuronJ in range(len(neural_network.layers[layerK])):
                afunc_derivs_vector.append(None)
            self.afunc_derivs_matrix.append(afunc_derivs_vector)

    def init_last_dq_dykj(self, guess: List[float], answer: List[float]):
        last_dq_dykj: List[float] = []  # initialize vector of all dq/dykj for last layer (k = len(self.layers)-1)
        # for each neuron in last layer
        for neuronJ in range(len(self.neural_network.layers[len(self.neural_network.layers)-1])):
            # calculate dq/dykj and append to vector
            last_dq_dykj.append(self.neural_network.loss_function.derivative(guess, answer, neuronJ))
        self.dq_dykj_matrix.append(last_dq_dykj)  # append calculated vector

    def get_dq_dykj(self, layerk: int, neuronj: int) -> float:
        if self.dq_dykj_matrix[layerk][neuronj] is None:  # if desired dq/dykj wasn't calculated yet
            self.calc_dq_dykj(layerk, neuronj)  # calculate this dq/dykj
        return self.dq_dykj_matrix[layerk][neuronj]  # return dq/dykj for given layer k and neuron j

    def calc_dq_dykj(self, layerk: int, neuronj: int):
        # calculate dq/dykj = Σ n = 1 to N (w(k+1)nj * ∂act_func(k+1)n(s(k+1)j)/∂s(k+1)j * dq/dy(k+1)n)
        sum_through_neurons: float = 0
        for neuronN in range(len(self.neural_network.layers[layerk+1])):
            addend: float = self.neural_network.layers[layerk+1][neuronN].weights[neuronj] *\
                            self.afunc_derivs_matrix[layerk+1][neuronN]
            addend = addend * self.get_dq_dykj(layerk+1, neuronN)
            sum_through_neurons = sum_through_neurons + addend
        self.dq_dykj_matrix[layerk][neuronj] = sum_through_neurons
