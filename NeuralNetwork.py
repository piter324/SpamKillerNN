from typing import List, Type, Tuple, Union
import Functions
import Neuron
from BackPropMatrices import BackPropMatrices
from TrainingSet import TrainingSet
from NetworkTester import NetworkTester
from MatrixMath import MatrixMath
from random import uniform
import pickle


class NeuralNetwork:
    def __init__(self, input_size: int, neurons_amount: List[int], weight_interval: Tuple[float, float],
                 act_functions: List[Type[Functions.FuncAbstract]], loss_function: Type[Functions.LossFuncAbstract]):
        # checking if parameters are legal
        assert weight_interval[0] < weight_interval[1]
        assert len(act_functions) == len(neurons_amount),\
            "Size of act_functions list(%d) doesn't equal layers_amount(%d):\nact_functions: %s"\
            % (len(act_functions), len(neurons_amount), act_functions)

        # initializing variables.
        self.loss_function = loss_function
        self.layers: List[List[Neuron.Neuron]] = []

        # initializing input layer with weights
        self.layers.append([])  # append new layer (input layer)
        for neuronN in range(neurons_amount[0]):  # add demanded amount of neurons to input layer
            weight_vector = []  # initialize weight vector
            for w in range(input_size+1):  # add input_size+1 weight values into weight_vector
                # get random value from given interval and append it to weight_vector
                weight_vector.append(uniform(weight_interval[0], weight_interval[1]))
            self.layers[0].append(Neuron.Neuron(weight_vector, act_functions[0]))  # append new neuron to input layer

        # initializing rest of layers with weights
        for layerK in range(1, len(neurons_amount), 1):  # add demanded amount of layers
            self.layers.append([])  # append new layer
            for neuronN in range(neurons_amount[layerK]):  # add demanded amount of neurons to current layer
                weight_vector = []  # initialize weight vector
                # add appropriate amount of weight values into weight_vector
                for w in range(neurons_amount[layerK-1]+1):
                    # get random value from given interval and append it to weight_vector
                    weight_vector.append(uniform(weight_interval[0], weight_interval[1]))
                # append new neuron to input layer
                self.layers[layerK].append(Neuron.Neuron(weight_vector, act_functions[layerK]))

    # \/\/ TRAINING STUFF \/\/
    def adjust_weights(self, adjust_matrix: List[List[List[float]]]):
        """Adjusts weights of all neurons in whole neural network."""

        for layerK in range(len(self.layers)):  # for each layer
            for neuronI in range(len(self.layers[layerK])):  # for each neuron in current layer
                # adjust weights using corresponding weight vector from adjust_matrix
                self.layers[layerK][neuronI].adjust_weights(adjust_matrix[layerK][neuronI])

    def examine_single_pair(self, input_data: List[float], answer: List[float]) -> List[List[List[float]]]:
        """
        Examines one training example
        :param input_data: input data for neural network
        :param answer: the correct answer for given input_data
        :return: matrix of loss functions derivatives in respect to all weights - dq/dwkij
        """

        back_prop_matrices = BackPropMatrices(self)  # create BackPropMatrices object used to make further calculations

        # process result and save derivatives of activation function and layers' outputs
        # result vector is called so due to its target meaning, before then it is "currently processed vector"
        result: List[float] = input_data.copy()  # take input_data that'll be gradually transformed into network output
        for layerK in range(len(self.layers)):  # for each layer
            result.append(1)
            back_prop_matrices.y.append(result.copy())  # save previous layer's output (or input if layerK == 0)
            iteration_result: List[float] = []  # initialize current layer's output vector
            for neuronI in range(len(self.layers[layerK])):  # for each neuron in current layer
                # get neuron's result
                neuron_output: Tuple[float, float] = self.layers[layerK][neuronI].process_input(result)
                iteration_result.append(neuron_output[0])  # append neuron's output to output vector
                # save derivative of neuron's activation function at point determined by neuron's sum
                back_prop_matrices.afunc_derivs_matrix[layerK][neuronI] =\
                    self.layers[layerK][neuronI].activationFunction.derivative(neuron_output[1])
            result = iteration_result  # substitute currently processed vector
        back_prop_matrices.y.append(result.copy())  # save last layer's output
        # calculate all dq/dykj for last layer (k = len(self.layers)-1)
        back_prop_matrices.init_last_dq_dykj(result.copy(), answer.copy())

        # calculate weights' derivatives matrix
        weight_derivs_matrix: List[List[List[float]]] = []  # matrix of all dq/dwkij (k - layer, i - neuron, j - weight)
        for layerK in range(len(self.layers)-1, -1, -1):  # for each layer but going backwards
            layers_derivs_vectors: List[List[float]] = []  # initialize list of derivatives for each i x j in current k
            for neuronI in range(len(self.layers[layerK])):  # for each neuron in current layer
                derivs_vector: List[float] = []  # initialize vector of derivatives for each j in current i
                for weightJ in range(len(self.layers[layerK][neuronI].weights)):  # for each weight in current neuron
                    # calculate dq/dwkij = ∂act_funcki(ski)/∂ski * y(k-1)j * dq/dyki
                    dqdw: float = back_prop_matrices.afunc_derivs_matrix[layerK][neuronI] *\
                                  back_prop_matrices.y[layerK][weightJ]
                    # above: it is layerK without -1 due to way of indexing - look y list in BackPropMatrices class
                    dqdw = dqdw * back_prop_matrices.get_dq_dykj(layerK, neuronI)
                    derivs_vector.append(dqdw)
                layers_derivs_vectors.append(derivs_vector)
            # prepend result because it goes backwards
            weight_derivs_matrix = [layers_derivs_vectors] + weight_derivs_matrix
        return weight_derivs_matrix

    def calc_gradientj(self, training_set: TrainingSet) -> List[List[List[float]]]:
        # gradient of J(w) function that we want to minimize
        gradient: List[List[List[float]]] = self.examine_single_pair(training_set.data[0], training_set.answers[0])
        for pairP in range(1, len(training_set.data), 1):
            gradient = MatrixMath.add3d(gradient, self.examine_single_pair(training_set.data[pairP],
                                                                           training_set.answers[pairP]))
        scalar: float = 1 / len(training_set.data)
        # multiply matrix by scalar
        gradient = MatrixMath.mul_scalar3d(gradient, scalar)
        return gradient

    def train(self, training_set: TrainingSet, learning_rate: float, learning_target: float, iterations_limit: int):
        assert 0 <= learning_target
        assert learning_rate > 0

        network_tester: NetworkTester = NetworkTester(self)
        test_result: float = network_tester.test(training_set)
        start_loss = test_result
        print("###STARTING TRAINING...###\nBefore training:\nTarget function: %s" % test_result)

        iteration: int = 0
        while test_result > learning_target and iteration < iterations_limit:
            gradient: List[List[List[float]]] = self.calc_gradientj(training_set)
            minus_beta_gradient = gradient.copy()
            # multiply matrix by scalar
            minus_beta_gradient = MatrixMath.mul_scalar3d(minus_beta_gradient, -learning_rate)

            self.adjust_weights(minus_beta_gradient)
            new_result = network_tester.test(training_set)
            if new_result > test_result:
                print("###Target function diverges (learning rate might be too big)###")
            test_result = new_result
            iteration = iteration + 1
            print("\nAfter { %d } iterations:\nTarget function: %s"
                  % (iteration, test_result))

        print("###END OF TRAINING###")
        print("\nTrained from loss function %s to %s\n" % (start_loss, test_result))

    def vtrain(self, training_set: TrainingSet, learning_rate: float,
               iterations_limit: int, validation_set: TrainingSet) -> Tuple[float, List[Tuple[float, float]]]:
        assert learning_rate > 0

        network_tester: NetworkTester = NetworkTester(self)
        test_result: float = network_tester.test(training_set)
        validation_result: float = network_tester.test(validation_set)
        start_loss = test_result
        stats: List[Tuple[float, float]] = [(start_loss, validation_result[0])]

        print("###STARTING TRAINING...###\nBefore training:\nTarget function: %s\nValidation target function: %s" %
              (test_result, validation_result))

        iteration: int = 0
        overfit_rating: float = 0
        while iteration < iterations_limit:
            gradient: List[List[List[float]]] = self.calc_gradientj(training_set)
            minus_beta_gradient = gradient.copy()
            # multiply matrix by scalar
            minus_beta_gradient = MatrixMath.mul_scalar3d(minus_beta_gradient, -learning_rate)

            print("ITERATION %d minus_beta_gradient: %s" % (iteration, minus_beta_gradient))
            self.adjust_weights(minus_beta_gradient)
            new_result = network_tester.test(training_set)
            new_validation_result = network_tester.test(validation_set)
            if iteration > iterations_limit/2:
                if new_validation_result > validation_result:
                    print("### OVERFITTING! ###")
                    overfit_rating += 1
                elif overfit_rating > 0:
                    overfit_rating -= 0.25
            if overfit_rating >= 4:
                test_result = new_result
                validation_result = new_validation_result
                break
            if new_result > test_result:
                print("###Target function diverges (learning rate might be too big)###")
            test_result = new_result
            validation_result = new_validation_result
            iteration += 1
            stats.append((test_result, validation_result))
            print("\nAfter { %d } iterations:\nTraining target function: %s\nValidation target function: %s"
                  % (iteration, test_result, validation_result))

        print("###END OF TRAINING###")
        print("\nTrained from loss function %s to %s\n" % (start_loss, test_result))
        return_tuple = (validation_result, stats)
        return return_tuple
    # /\/\ TRAINING STUFF /\/\

    def make_guess(self, input_vector: List[float]) -> List[float]:
        # Checking if input is legal.
        assert len(input_vector) == len(self.layers[0][0].weights)-1,\
            "Size of input_vector is (%d), but layer (0) expects input of size (%d+1):" \
            "\ninput_vector: %s\nself.layers[0][0].weights: %s" %\
            (len(input_vector), len(self.layers[0][0].weights)-1, input_vector, self.layers[0][0].weights)

        # Start of processing.
        result: List[float] = input_vector.copy()
        for layerK in range(len(self.layers)):
            result.append(1)
            iteration_result: List[float] = []
            for neuronI in range(len(self.layers[layerK])):
                iteration_result.append(self.layers[layerK][neuronI].process_input(result)[0])
            result = iteration_result
        return result

    def get_weights(self) -> List[List[List[float]]]:
        weight_matrix: List[List[List[float]]] = []
        for layerK in self.layers:
            layer_matrix: List[List[float]] = []
            for neuronI in layerK:
                layer_matrix.append(neuronI.weights.copy())
            weight_matrix.append(layer_matrix)
        return weight_matrix

    def set_weights(self, weight_matrix: List[List[List[float]]]):
        for layerK in range(len(self.layers)):
            for neuronI in range(len(self.layers[layerK])):
                self.layers[layerK][neuronI].weights = weight_matrix[layerK][neuronI].copy()

    def save(self, file_name: Union[str, None]):
        if file_name is None:
            file_name = "network" + str(id(self)) + ".pkl"
        with open(file_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_name: str):
        with open(file_name, 'rb') as input:
            return pickle.load(input)
