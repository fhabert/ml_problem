# Task 1 Implementation
import math 

class Node:
    def __init__(self, no_input):
        self.val = None
        self.bias_weight = None 
        self.af = None
        self.weights = [1 for i in range(no_input)]
        self.delta = None
    
    def set_val(self, val):
        # only for input neurons
        self.val = val

    def set_bias_weight(self, bias_weight):
        # only for output neurons
        self.bias_weight = bias_weight
    
    def set_activation_function(self, activation_function):
        # only for output neurons
        self.af = activation_function
    
    def set_delta(self, delta):
        # only for backpropagation
        self.delta = delta
    
    def print_node(self):
        string =  "Value: " + str(self.val) + ", Bias: " + str(self.bias_weight) + ", Activation Function: " + str(self.af) + ", Node: " + str(self.weights)
        return string
    
    def set_weights(self, weights):
        self.weights = weights

class NN:
    def __init__(self, structure, bias_activation):
        # structure e.g. [1, 3, 1]
        # Each layer is a list of nodes
        self.layers = self.construct_layers(structure, bias_activation)

    def set_inputs_in_layer(self, inputs):
        # set values for input layer only
        input_layer = self.layers[0]
        for i in range(len(input_layer)):
            input_layer[i].set_val(inputs[i])

    def foward_pass_layer(self, layer_index):
        # cannot carry this out on input layer
        if layer_index == 0:
            return
        input_nodes = self.layers[layer_index-1]
        for output in self.layers[layer_index]:
            sum = 0
            w_index = 0
            for input in input_nodes:
                sum += input.val * output.weights[w_index]
                w_index += 1
            output.set_val(output.af(sum  + output.bias_weight))
    
    def print_layer(self, layer_index):
        for node in self.layers[layer_index]:
            print ("[" + node.print_node() + "]")
    
    def construct_layers(self, structure, bias_activation):
        # can take in bias/activation each as a list of lists
        # eg for 1-3-1 structure [[[b11,a11], [b12,a12], [b13,a13]], [[b21,a21]]] 

        layers = []
        for i in range(len(structure)):
            if i == 0:
                # input layer nodes have 0 inputs
                nodes = [Node(0) for x in range(structure[i])]
            else:
                nodes = [ Node(structure[i-1]) for x in range(structure[i])]

            # set biases and activation functions
            if i != 0:
                bias_activation_pair = bias_activation[i-1]
                for i in range(len(nodes)):
                    nodes[i].set_bias_weight(bias_activation_pair[i][0])
                    nodes[i].set_activation_function(bias_activation_pair[i][1])
    
            layers.append(nodes)
        return layers
    
    def print_nn(self):
        for i in range(len(self.layers)):
            print ("**********")
            print ("LAYER")
            print ("**********")
            self.print_layer(i)
    
    def foward_pass_nn(self, inputs):
        for i in range(len(self.layers)):
            # carry out forward pass on one layer to update output val
            if i == 0:
                self.set_inputs_in_layer(inputs)
            else:
                self.foward_pass_layer(i)
    
    def get_output(self):
        output_vals = []
        for output in self.layers[-1]:
            output_vals.append(output.val)


class Backpropagation:
    def __init__(self, nn, desired, rate):
        self.layers = nn.layers
        self.output = nn.get_output()
        self.desired = desired
        self.rate = rate
    
    def calc_output_delta(self):
        output_neurons = self.layers[-1]
        for i in range(len(output_neurons)):
            output = output_neurons[i]
            af = output.af
            delta = (self.desired[i]-output.val)*derivative(output.val,af)
            output.set_delta(delta)
    
    def update_weights(self, layer_index):
        front_neurons = self.layers[layer_index]
        back_neurons = self.layers[layer_index-1]
        for i in range(len(front_neurons)):
            for j in range(len(back_neurons)):
                neuron = front_neurons[i]
                w_old = neuron.weights[j]
                prev_neuron_val = back_neurons[j].val
                neuron.weights[j] = w_old + self.rate*neuron.delta*prev_neuron_val

    
    def calc_hidden_delta(self, layer_index):
        output_neurons = self.layers[layer_index+1] # should already have delta values
        curr_neurons = self.layers[layer_index]
        for i in range(len(curr_neurons)):
            curr_neuron = curr_neurons[i]
            sum_delta_weights = 0
            for j in range(len(output_neurons)):
                neuron = output_neurons[j]
                sum_delta_weights += neuron.delta*neuron.weights[i]
            curr_neuron.delta = sum_delta_weights*derivative(curr_neuron.val, curr_neuron.af)
    
    def update_bias_weight(self, layer_index):
        curr_neurons = self.layers[layer_index]
        for i in range(len(curr_neurons)):
            neuron = curr_neurons[i]
            old_bias_weight = neuron.bias_weight
            neuron.bias_weight = old_bias_weight + self.rate*neuron.delta


    def backpropagation(self):
        self.calc_output_delta()
        self.update_weights(-1)
        self.update_bias_weight(-1)
        for i in range(len(self.layers)-2, 0, -1):
            self.calc_hidden_delta(i)
            self.update_weights(i)
            self.update_bias_weight(i)
    
    def print_nn(self):
        for i in range(len(self.layers)):
            print ("**********")
            print ("LAYER")
            print ("**********")
            self.print_layer(i)
    
    def print_layer(self, layer_index):
        for node in self.layers[layer_index]:
            print ("[" + node.print_node() + "]")
        

# Activation functions

def sigmoid(input):
    return 1/(1 + math.e**(-input))

def tanh(input):
    return math.tanh(input)

def linear(input):
    return input

def relu(input):
    return max(0, input)

def leaky_relu(input):
    return max(0.01*input, input)

def derivative(input, function):
    # NOTE input is X = af(v) 
    if function == sigmoid:
        return (1-input)*input
    elif function == tanh:
        return 1-input**2
    elif function == linear:
        return 1
    elif function == relu:
        if input < 0:
            return 0
        else:
            return 1
    elif function == leaky_relu:
        if input < 0:
            return 0.01
        else:
            return 1


structure = [1,3,2]
# [[[b11,a11], [b12,a12], [b13,a13]], [[b21,a21]]] 
bias_activation = [[[2 ,tanh], [4,tanh], [3,tanh]], [[2,tanh], [6, tanh]]] 
neural_network = NN(structure, bias_activation)

neural_network.layers[1][0].weights = [1]
neural_network.layers[1][1].weights = [2]
neural_network.layers[1][2].weights = [3]
neural_network.layers[2][0].weights = [5, 4, 8]
neural_network.layers[2][1].weights = [3, 6, 2]


neural_network.foward_pass_nn([-1])
neural_network.print_nn()

backprop = Backpropagation(neural_network, [20, 10], 0.01)
backprop.backpropagation()
backprop.print_nn()

structure = [1,3,2]
bias_activation = [[[2 ,sigmoid], [3, tanh], [4, sigmoid]], [[-3,tanh], [0.5,tanh]]] 
neural_network = NN(structure, bias_activation)
neural_network.foward_pass_nn([1])
neural_network.print_nn()
