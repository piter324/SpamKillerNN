from typing import List, Type, Tuple
import Functions
import Neuron


class NeuralNetwork:
    def __init__(self, layers_amount: int, neurons_amount: List[int], weight_matrix: List[List[float]],
                 act_functions: List[Type[Functions.FuncAbstract]], loss_function: Type[Functions.LossFuncAbstract]):
        # \/ \/Checking if parameters are legal.\/ \/
        assert layers_amount > 0, "layers_amount(%d) is not greater than 0" % layers_amount
        assert len(neurons_amount) == layers_amount,\
            "Size of neurons_amount list(%d) doesn't equal layers_amount(%d):\nneurons_amount: %s"\
            % (len(neurons_amount), layers_amount, neurons_amount)
        assert len(weight_matrix) == layers_amount,\
            "Size of weight_matrix list(%d) doesn't equal layer_amount(%d):\nweight_matrix: %s"\
            % (len(weight_matrix), layers_amount, weight_matrix)
        assert len(act_functions) == layers_amount,\
            "Size of act_functions list(%d) doesn't equal layers_amount(%d):\nact_functions: %s"\
            % (len(act_functions), layers_amount, act_functions)
        for k in range(1, layers_amount):
            assert neurons_amount[k-1] == len(weight_matrix[k])-1,\
                "Size of layer's (%d) output is (%d), but layer (%d) expects input of size (%d+1):" \
                "\nneurons_amount[%d]: %d\nlen(weight_matrix[%d]): %d" %\
                (k-1, neurons_amount[k-1], k, len(weight_matrix[k])-1,
                 k-1, neurons_amount[k-1], k, len(weight_matrix[k]))
        # /\ /\Checking if parameters are legal./\ /\
        # Initializing variables.
        self.loss_function = loss_function
        self.layers: List[List[Neuron.Neuron]] = []
        for layerK in range(layers_amount):
            self.layers.append([])
            for neuronN in range(neurons_amount[layerK]):
                self.layers[layerK].append(Neuron.Neuron(weight_matrix[layerK], act_functions[layerK]))
        #for k in range(layers_amount):  # TODO do wywalenia
            #print("LAYER %d" % k)
            #for n in range(neurons_amount[k]):
                #print(self.layers[k][n].activationFunction)
                #print(self.layers[k][n].weights)

    # \/\/ TRAINING STUFF \/\/
    def adjust(self, adjust_matrix: List[List[List[float]]]):
        # Checking if adjust_matrix is legal.
        assert len(adjust_matrix) == len(self.layers)
        for k in range(len(self.layers)):
            assert len(adjust_matrix[k]) == len(self.layers[k][0].weights)
        # Adjusting weights.
        for layerK in range(len(self.layers)):
            for neuronN in range(len(self.layers[layerK])):
                self.layers[layerK][neuronN].adjust_weights(adjust_matrix[layerK][neuronN])

    def examine_single_pair(self, pair: Tuple[List[float], List[float]]) -> List[List[List[float]]]:
        s: List[List[float]] = []  # 2D matrix of all sums, s[k][i]: k - layer index, i - neuron index
        # TODO ^ maybe i can calculate ∂act_func(ski)/∂ski during processing result instead of saving only sums
        y: List[List[float]] = []  # 2D matrix of all layers' outputs, y[k]: k-th layer output

        # process result and save to s and y

        derivatives_matrix: List[List[List[float]]] = []

        # TODO
        # we might not want this loop to look like this
        for k in range(len(self.layers)):  # k-th layer
            layers_derivs_vectors: List[List[float]] = []
            for i in range(len(self.layers[k])):  # i - i-th neuron in k-th layer
                deriv_vector: List[float] = []
                for j in range(len(self.layers[k][i].weights)):  # j - weight index of i-th neuron in k-th layer
                    dqdw: float = 0

                    # calculate dqdw

                    deriv_vector.append(dqdw)
                layers_derivs_vectors.append(deriv_vector)
            derivatives_matrix.append(layers_derivs_vectors)
        return derivatives_matrix

    def train(self, training_set: list, learning_rate: float):  # TODO
        # TODO
        pass

    def kfold_train(self):  # TODO
        pass
    # /\/\ TRAINING STUFF /\/\

    def guess(self, input_vector: List[float]) -> List[float]:
        # Checking if input is legal.
        assert len(input_vector) == len(self.layers[0][0].weights)-1,\
            "Size of input_vector is (%d), but layer (0) expects input of size (%d+1):" \
            "\ninput_vector: %s\nself.layers[0][0].weights: %s" %\
            (len(input_vector), len(self.layers[0][0].weights)-1, input_vector, self.layers[0][0].weights)
        # Start of processing.
        result: List[float] = input_vector.copy()
        for k in range(len(self.layers)):
            iteration_result: List[float] = []
            for n in range(len(self.layers[k])):
                iteration_result.append(self.layers[k][n].process_input(result))
            result = iteration_result
        print("Input %s gave result: %s" % (input_vector, result))
        return result
