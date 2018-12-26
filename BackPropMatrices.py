from typing import List, Union


class BackPropMatrices:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.dq_dykj_matrix: List[List[Union[float, None]]] = []  # dq/dykj
        self.afunc_derivs_matrix: List[List[float]] = []  # ∂act_func(skj)/∂skj k-th layer, j-th neuron
        self.y: List[List[float]] = []  # y[0] - input, y[k] k=1,2,... - output of k-1-th layer

        # init dq_dykj_matrix with Nones
        for layerK in range(len(neural_network.layers)-1):
            dq_dykj_vector: List[None] = []
            for neuronJ in range(len(neural_network.layers[layerK])):
                dq_dykj_vector.append(None)
            self.dq_dykj_matrix.append(dq_dykj_vector)

        # TODO might not be needed as well as assertion in set_deriv
        # init afunc_derivs_matrix
        for layerK in range(len(neural_network.layers)):
            afunc_derivs_vector: List[None] = []
            for neuronJ in range(len(neural_network.layers[layerK])):
                afunc_derivs_vector.append(None)
            self.afunc_derivs_matrix.append(afunc_derivs_vector)

    def init_last_dq_dykj(self, guess: List[float], answer: List[float]):
        last_dq_dykj: List[float] = []
        for neuronJ in range(len(self.neural_network.layers[len(self.neural_network.layers)-1])):
            last_dq_dykj.append(self.neural_network.loss_function.derivative(guess, answer, neuronJ))
        self.dq_dykj_matrix.append(last_dq_dykj)

    # def set_deriv(self, layerk: int, neuroni: int, value: float):
    #     assert self.afunc_derivs_matrix[layerk][neuroni] is None,\
    #         "Tried to set derivative (%d, %d), but it was already set!" % (layerk, neuroni)
    #     self.afunc_derivs_matrix[layerk][neuroni] = value

    def get_dq_dykj(self, layerk: int, neuronj: int) -> float:
        if self.dq_dykj_matrix[layerk][neuronj] is None:
            self.calc_dq_dykj(layerk, neuronj)
        return self.dq_dykj_matrix[layerk][neuronj]

    def calc_dq_dykj(self, layerk: int, neuronj: int):
        sum_through_neurons: float = 0
        for neuronN in range(len(self.neural_network.layers[layerk+1])):
            addend: float = self.neural_network.layers[layerk+1][neuronN].weights[neuronj] *\
                            self.afunc_derivs_matrix[layerk+1][neuronN]
            addend = addend * self.get_dq_dykj(layerk+1, neuronN)
            sum_through_neurons = sum_through_neurons + addend
        self.dq_dykj_matrix[layerk][neuronj] = sum_through_neurons
