# Task 1 Implementation
import math 
import random
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import time

class Node:
    def __init__(self, no_input):
        self.val = None
        self.bias_weight = None 
        self.af = None
        self.weights = [random.randint(0,1) for i in range(no_input)]
        self.delta = None
    
    def set_val(self, val):
        # Set value for input neuron
        self.val = val

    def set_bias_weight(self, bias_weight):
        # Set bias weight
        self.bias_weight = bias_weight
    
    def set_activation_function(self, activation_function):
        # Set activation function 
        self.af = activation_function
    
    def set_delta(self, delta):
        # Set delta value for backpropagation
        self.delta = delta
    
    def set_weights(self, weights):
        # Set a list of weights from connections
        # coming into neuron
        self.weights = weights
    
    def print_node(self):
        # Print single node for debugging
        string =  "Value: " + str(self.val) + ", Bias: " + str(self.bias_weight) + ", Activation Function: " + str(self.af) + ", Node: " + str(self.weights)
        return string

class NN:
    def __init__(self, structure, bias_activation, rate):
        # structure e.g. [1, 3, 1]
        # Each layer is a list of nodes
        self.layers = self.construct_layers(structure, bias_activation)
        # List of output values to comapre with desired list for backprop
        self.output = self.get_output()
        # Desired values correspond to each output neuron and
        # are initialised just before backpropagation
        self.desired = None
        self.rate = rate

    def set_inputs_in_layer(self, inputs):
        # set values for input layer only
        input_layer = self.layers[0]
        for i in range(len(input_layer)):
            input_layer[i].set_val(inputs[i])

    def foward_pass_layer(self, layer_index):
        # Forward pass on a single layer
        if layer_index == 0:
            # Cannot carry this out on input layer
            return
        input_nodes = self.layers[layer_index-1]
        for output in self.layers[layer_index]:
            sum = 0
            w_index = 0
            for input in input_nodes:
                sum += input.val * output.weights[w_index]
                w_index += 1
            output.set_val(output.af(sum  + output.bias_weight))
    
    def foward_pass_nn(self, inputs):
        # Iterates through the whole nn to carry
        # out forward pass per layer
        for i in range(len(self.layers)):
            # Carry out forward pass on one layer to update output val
            if i == 0:
                self.set_inputs_in_layer(inputs)
            else:
                self.foward_pass_layer(i)
    
    def construct_layers(self, structure, bias_activation):
        # Builds the nn with specificed structure, bias and activation functions
        # Takes in bias/activation each as a list of lists
        # eg for 1-3-1 structure [[[b11,a11], [b12,a12], [b13,a13]], [[b21,a21]]] 

        layers = []
        for i in range(len(structure)):
            if i == 0:
                # Input layer nodes have 0 inputs
                nodes = [Node(0) for x in range(structure[i])]
            else:
                nodes = [ Node(structure[i-1]) for x in range(structure[i])]

            # Set biases and activation functions
            if i != 0:
                bias_activation_pair = bias_activation[i-1]
                for i in range(len(nodes)):
                    nodes[i].set_bias_weight(bias_activation_pair[i][0])
                    nodes[i].set_activation_function(bias_activation_pair[i][1])
    
            layers.append(nodes)
        return layers

    def print_layer(self, layer_index):
        # Print a single layer for debugging
        for node in self.layers[layer_index]:
            print ("[" + node.print_node() + "]")
    
    def print_nn(self):
        # Print the whole nn for debugging
        for i in range(len(self.layers)):
            print ("**********")
            print ("LAYER")
            print ("**********")
            self.print_layer(i)

    def get_output(self):
        # Go through every output neuron from
        # current nn to collect list of outputs
        output_vals = []
        for output in self.layers[-1]:
            output_vals.append(output.val)
    
    def set_desired(self, desired):
        # Set a list of designed outputs from nn
        # before backpropagation
        self.desired = desired
    
    def calc_output_delta(self):
        # Get delta values for neurons in output layer
        output_neurons = self.layers[-1]
        for i in range(len(output_neurons)):
            output = output_neurons[i]
            af = output.af
            delta = (self.desired[i]-output.val)*derivative(output.val,af)
            output.set_delta(delta)
    
    def update_weights(self, layer_index):
        # Update weights in each layer for backprop
        front_neurons = self.layers[layer_index]
        back_neurons = self.layers[layer_index-1]
        for i in range(len(front_neurons)):
            for j in range(len(back_neurons)):
                neuron = front_neurons[i]
                w_old = neuron.weights[j]
                prev_neuron_val = back_neurons[j].val
                neuron.weights[j] = w_old + self.rate*neuron.delta*prev_neuron_val
    
    def calc_hidden_delta(self, layer_index):
        # Calculate delta values for neurons in any layer
        # apart from output layer.
        output_neurons = self.layers[layer_index+1] 
        curr_neurons = self.layers[layer_index]
        for i in range(len(curr_neurons)):
            curr_neuron = curr_neurons[i]
            sum_delta_weights = 0
            for j in range(len(output_neurons)):
                neuron = output_neurons[j]
                sum_delta_weights += neuron.delta*neuron.weights[i]
            curr_neuron.delta = sum_delta_weights*derivative(curr_neuron.val, curr_neuron.af)
    
    def update_bias_weight(self, layer_index):
        # Update bias weights per layer in each layer for backprop
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


def set_dataframe(df):
    amount_train = round(0.8*len(df))
    titles = df.iloc[:, 0].unique()
    one_hot_titles = []
    for i in range(len(titles)):
        category = [0.2 if i != j else 0.8 for j in range(len(titles))]
        one_hot_titles.append(category)
    l = len(df)
    df_randomized = [list(df.iloc[i]) for i in range(l)]
    for _ in range(len(df_randomized)):
        i = random.randint(0,l-1)
        j = random.randint(0,l-1)
        df_randomized[i], df_randomized[j] = df_randomized[j], df_randomized[i]
    
    for i in range(len(df_randomized)):
        if df_randomized[i][0] == "Iris-virginica":
            df_randomized[i][0] = one_hot_titles[0]
        elif df_randomized[i][0] == "Iris-setosa":
            df_randomized[i][0] = one_hot_titles[1]
        else:
            df_randomized[i][0] = one_hot_titles[2]
    new_df = np.array(df_randomized,dtype=object)
    features_train = new_df[:, 1:][:amount_train]
    features_labels = new_df[:, 0][:amount_train]
    validation_train = new_df[:, 1:][amount_train:]
    validation_labels = new_df[:, 0][amount_train:]
    return features_train, features_labels, validation_train, validation_labels


# set up neural network

def construct_nn(func, lr):
    structure = [4,5,3,3]
    # [[[b11,a11], [b12,a12], [b13,a13]], [[b21,a21]]]
    bias_activation = []
    counter = 0
    for item in structure:
        if counter != 0 and counter != len(structure)-1:
            inner_list = [[random.randint(0,1),func[0]] for _ in range(item)]
            bias_activation.append(inner_list)
        elif counter != 0 and counter == len(structure)-1:
            inner_list = [[random.randint(0,1),func[1]] for _ in range(item)]
            bias_activation.append(inner_list)
        counter += 1
        
    neural_network = NN(structure, bias_activation, lr)
    return neural_network

def visualize_result(x, y, predection):
    plt.scatter(x, y)
    plt.plot(x, predection, alpha=0.7, color="b")
    plt.show()
    pass

def single_epoch(neural_network, x, desired):
    for i in range(len(x)):
        neural_network.foward_pass_nn(x[i])
        neural_network.set_desired(desired[i])
        neural_network.backpropagation()
    return neural_network

def get_output(neural_network, x):
    y = []
    for i in range(len(x)):
        inner_list = []
        neural_network.foward_pass_nn(list(x[i]))
        for j in range(len(neural_network.layers[-1])):
            inner_list.append(neural_network.layers[-1][j].val)
        # index = np.argmax(inner_list)
        y.append(inner_list)    
    return y

test_activation_functions = [[tanh, linear], [tanh, tanh]]
test_combinaison = 1

dataframe = pd.read_csv("./dataset_iris.csv", sep=";", encoding="utf-8")
features_train, features_labels, validation_train, validation_labels = set_dataframe(dataframe)

x = np.arange(-1, 1, 0.05)
y = []
norm_noise = np.random.normal(0,0.02,len(x))
for item in x:
    rand_index = random.randint(0, len(x)-1)
    result = (item**3)*0.8 + 0.3*(item**2) - 0.4*item + norm_noise[rand_index]
    y.append(result)

no_epochs = 10000
learning_rates = [0.005, 0.008, 0.01, 0.013, 0.016, 0.018, 0.021]
results_accuracy = []
for rate in learning_rates:
    start_time = time.time()
    for i in range(no_epochs):
        neural_network = construct_nn(test_activation_functions[test_combinaison], rate)
        nn_trained = single_epoch(neural_network, features_train, features_labels)

    score = 0
    for i in range(len(validation_train)):
        decode = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        tested_output = get_output(nn_trained, [validation_train[i]])
        max_value = 0
        index_output = 0
        for j in range(len(tested_output[0])):
            if tested_output[0][j] > max_value:
                max_value = tested_output[0][j]
                index_output = j
        output_iris = decode[index_output]
        max_value = 0
        index_label = 0
        for j in range(len(validation_labels[i])):
            if validation_labels[i][j] > max_value:
                max_value = validation_labels[i][j]
                index_label = j
        actual_iris = decode[index_label]
        if output_iris == actual_iris:
            score += 1
    end_time = time.time()
    total_time = end_time - start_time
    # print(score)
    # print(len(validation_labels))
    norm_score = score/len(validation_labels)*100
    print("Accuracy test:", norm_score, "%")
    results_accuracy.append((round(norm_score,2), round(total_time,4)))
    # tested_output = get_output(nn_trained, validation_train[:5])
    # print("Output results:", tested_output)
    # print("Actual output:", validation_labels[:5])

print(results_accuracy)

# test1 = [(93.33, 23.6026), (90.0, 23.5905), (90.0, 23.39483), (90.0, 23.6013), (90.0, 23.6967), (90.0, 23.6750), (20.0, 23.7686)]
# test2 = [(100.0, 24.0569), (100.0, 23.9315), (100.0, 23.7617), (100.0, 23.6321), (100.0, 23.6589), (100.0, 23.8701), (70.0, 23.9306)]
# learning_rates = [0.005, 0.008, 0.01, 0.013, 0.016, 0.018, 0.021]
# mean_test = np.add(test1, test2) / 2
# print(mean_test)
# fig, ax1 = plt.subplots() 

# ax1.set_title("Learning Rates analysis in terms of accuracy and time \n for execution over 2 trials")
# ax1.set_xlabel('Learning Rates') 
# ax1.set_ylabel('Accuracy test (%)', color = 'red') 
# ax1.plot(learning_rates, mean_test[:, 0], color = 'red') 
# ax1.tick_params(axis ='y', labelcolor = 'red') 
  
# # Adding Twin Axes

# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Time taken for execution (s)', color = 'blue') 
# ax2.plot(learning_rates,  mean_test[:, 1], color = 'blue') 
# ax2.tick_params(axis ='y', labelcolor = 'blue') 
# fig.tight_layout()
# # Show plot

# plt.show()










# results_old_version = [99.66, 93.33, 93.33, 93.33, 53,33, 40, 100, 96.66, 30, 66]
# results_new_version = [86.66, 43.33, 53.33, 100, 43.3, 96.6, 36.6, 100, 83.33, 40]
# x = [i for i in range(len(results_new_version))]
# plt.bar(x, results_new_version, color="g", alpha=0.5)
# plt.ylabel("Accuracy (%)")
# plt.xlabel("Trials")
# plt.title(f"Newer version: mean of {round(np.mean(results_new_version), 2)}%")
# plt.show()