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
        self.weights = [random.uniform(-1,1) for i in range(no_input)]
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
        # Set a ist of weights from connections
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
        # E.g. for Setosa the encoding would be [0.8, 0.2, 0.2]
        category = [0.2 if i != j else 0.8 for j in range(len(titles))]
        one_hot_titles.append(category)

    # Swapping rows of csv file
    l = len(df)
    df_randomized = [list(df.iloc[i]) for i in range(l)]
    for _ in range(len(df_randomized)):
        i = random.randint(0,l-1)
        j = random.randint(0,l-1)
        df_randomized[i], df_randomized[j] = df_randomized[j], df_randomized[i]

    # Allocating one hot titles
    for i in range(len(df_randomized)):
        if df_randomized[i][0] == "Iris-virginica":
            df_randomized[i][0] = one_hot_titles[0]
        elif df_randomized[i][0] == "Iris-setosa":
            df_randomized[i][0] = one_hot_titles[1]
        else:
            df_randomized[i][0] = one_hot_titles[2]
    
    # Split data frame into training and test
    new_df = np.array(df_randomized,dtype=object)
    training_features = new_df[:, 1:][:amount_train]
    training_labels = new_df[:, 0][:amount_train]
    test_features = new_df[:, 1:][amount_train:]
    test_labels = new_df[:, 0][amount_train:]
    return training_features, training_labels, test_features, test_labels


# set up neural network

def construct_nn(func, learning_r):
    structure = [4,5,3,3]

    # Set bias weights and activation functions for nn
    bias_activation = []
    counter = 0
    for item in structure:
        if counter != 0 and counter != len(structure)-1:
            inner_list = [[random.uniform(-1,1),func[0]] for _ in range(item)]
            bias_activation.append(inner_list)
        elif counter != 0 and counter == len(structure)-1:
            inner_list = [[random.uniform(-1,1),func[1]] for _ in range(item)]
            bias_activation.append(inner_list)
        counter += 1
        
    # Create nn object
    neural_network = NN(structure, bias_activation, learning_r)
    return neural_network

def single_epoch(neural_network, x, desired):
    for i in range(len(x)):
        neural_network.foward_pass_nn(list(x[i]))
        neural_network.set_desired(list(desired[i]))
        neural_network.backpropagation()
    return neural_network

def get_output(neural_network, x):
    # Carries out forward pass of each sample
    # in input matrix x to get a list of ouput values
    # from trained nn
    y = []
    for i in range(len(x)):
        inner_list = []
        neural_network.foward_pass_nn(list(x[i]))
        for j in range(len(neural_network.layers[-1])):
            inner_list.append(neural_network.layers[-1][j].val)
        # index = np.argmax(inner_list)
        y.append(inner_list)    
    return y

def run_single_test(no_epochs, learning_r):
    # Contruct nn with chosen activation function combinations
    test_activation_functions = [[tanh, linear], [tanh, tanh], [sigmoid, leaky_relu], [leaky_relu, sigmoid]]
    af_combintation = 2
    neural_network = construct_nn(test_activation_functions[af_combintation], learning_r)

    # Read from csv file to extract training and test data
    dataframe = pd.read_csv("./dataset_iris.csv", sep=";", encoding="utf-8")
    training_features, training_labels, test_features, test_labels = set_dataframe(dataframe)

    # Train nn with training features and labels
    for i in range(no_epochs):
        nn_trained = single_epoch(neural_network, training_features, training_labels)

    # Test nn model with test features and labels
    score = 0
    test_output = get_output(nn_trained, test_features)
    for i in range(len(test_output)):
        pred = np.argmax(test_output[i])
        actual = np.argmax(test_labels[i])
        if pred == actual:
            score += 1

    accuracy = score/len(test_labels)*100
    # Print accuracy
    # print("Accuracy test:", accuracy, "%")
    return accuracy

epochs = 5000
no_tests = 6
accuracy = []
# epochs_rate = [x for x in range(500, 10000, 2000)]
learning_rates = np.arange(0.005, 0.025, 0.003)
for i in range(no_tests):
    dif_func_results = []
    for lr in learning_rates:
        start_time = time.time()
        percentage = run_single_test(epochs, lr)
        end_time = time.time()
        elapsed_t = end_time - start_time
        dif_func_results.append((percentage, elapsed_t))
    accuracy.append(dif_func_results)
    print("done")
#     print(dif_func_results)
# print(accuracy)

results = np.array(accuracy)
learning_rates = [0.005, 0.008, 0.01, 0.013, 0.016, 0.019, 0.021]
total_ar_per = []
total_duration = []
for i in range(len(results[0])):
    inner_list = results[:, i]
    sum_dur = 0
    sum_per = 0
    for item in inner_list:
        sum_per += item[0]
        sum_dur += item[1]
    total_ar_per.append(sum_per/len(results))
    total_duration.append(sum_dur/len(results))

fig, ax1 = plt.subplots() 
ax1.set_title(f"Learning Rates analysis in terms of accuracy and time taken \n for execution over {no_tests} trials and 5,000 epochs")
ax1.set_xlabel('Learning Rates') 
ax1.set_ylabel('Accuracy test (%)', color = 'red') 
ax1.plot(learning_rates, total_ar_per, color = 'red', alpha=0.6) 
ax1.tick_params(axis ='y', labelcolor = 'red') 
ax2 = ax1.twinx() 
ax2.set_ylabel('Time taken for execution (s)', color = 'blue') 
ax2.plot(learning_rates, total_duration, color = 'blue', alpha=0.6) 
ax2.tick_params(axis ='y', labelcolor = 'blue') 
fig.tight_layout()
plt.show()


# results = np.array(accuracy)
# total_ar = []
# for i in range(len(results[0])):
#     total_ar.append(np.sum(results[:, i])/len(results))
# plt.bar(epochs_rate, total_ar, color="b", alpha=0.6, width = 200)
# plt.ylim([50,100])
# plt.xlabel("Number of epochs")
# plt.ylabel("Accuracy (%)")
# plt.title("Accuracy of number of Epochs analysis \n for a learning rate of 0.01 over three trials")
# plt.show()