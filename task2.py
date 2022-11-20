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
            output.set_val(output.af(sum + output.bias_weight))
    
    def print_layer(self, layer_index):
        for node in self.layers[layer_index]:
            # print ("[" + node.print_node() + "]")
            pass
    
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
            # print ("**********")
            # print ("LAYER")
            # print ("**********")
            # self.print_layer(i)
            pass
    
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
        desired = self.desired[0]
        for i in range(len(output_neurons)):
            output = output_neurons[i]
            af = output.af
            delta = (desired[i]-output.val)*derivative(output.val,af)
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
            # print ("**********")
            # print ("LAYER")
            # print ("**********")
            # self.print_layer(i)
            pass
    
    def print_layer(self, layer_index):
        for node in self.layers[layer_index]:
            # print ("[" + node.print_node() + "]")
            pass
    
   
        

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
    amount_train = round(0.7*len(df))
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

def construct_nn(func):
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
        
    neural_network = NN(structure, bias_activation)
    return neural_network

def visualize_result(x, y, predection):
    plt.scatter(x, y)
    plt.plot(x, predection, alpha=0.7, color="b")
    plt.show()
    pass

def single_epoch(neural_network, x, rate, desired):
    for i in range(len(x)):
        neural_network.foward_pass_nn(list(x[i]))
        backprop = Backpropagation(neural_network, [desired[i]], rate)
        backprop.backpropagation()
        neural_network.layers = backprop.layers
        neural_network.print_nn()

    backprop.print_nn()
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

# test_activation_functions = [[tanh, linear], [tanh, tanh], [sigmoid, leaky_relu], [leaky_relu, sigmoid]]
# test_combinaison = 1
# neural_network = construct_nn(test_activation_functions[test_combinaison])

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
test_activation_functions = [[tanh, linear], [tanh, tanh], [sigmoid, leaky_relu], [leaky_relu, sigmoid]]
total_accuracy = []
neural_network = construct_nn([sigmoid, leaky_relu])
for i in range(no_epochs):
    nn_trained = single_epoch(neural_network, features_train, 0.01, features_labels)
score = 0
for i in range(len(validation_train)):
    decode = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
    tested_output = get_output(nn_trained, [validation_train[i]])
    print(tested_output)
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
norm_score = score/len(validation_labels)*100
total_accuracy.append(norm_score)
print("Accuracy test:", norm_score, "%")

print(total_accuracy)

#### Learning rate different results

# test1 = [(100.0, 23.8463), (96.67, 23.7034), (96.67, 23.8239), (96.67, 23.6394), (96.67, 23.8041), (96.67, 24.0441), (96.67, 23.7885)]
# test2 = [(50.0, 23.7487), (50.0, 24.7385), (96.67, 24.6076), (93.33, 34.806), (93.33, 24.0909), (93.33, 23.8515), (93.33, 24.239)]
# test3 = [(96.67, 23.6875), (93.33, 24.8601), (93.33, 24.4612), (93.33, 25.014), (90.0, 24.6902), (96.67, 24.3055), (90.0, 24.3149)]
# learning_rates = [0.005, 0.008, 0.01, 0.013, 0.016, 0.019, 0.021]

# mean_test = np.add(test1, test2)
# mean_total_test =  np.add(mean_test, test3) / 3
# print(mean_test)
# fig, ax1 = plt.subplots() 

# ax1.set_title("Learning Rates analysis in terms of accuracy and time taken \n for execution over 3 trials and 10,000 epochs")
# ax1.set_xlabel('Learning Rates') 
# ax1.set_ylabel('Accuracy test (%)', color = 'red') 
# ax1.plot(learning_rates, mean_total_test[:, 0], color = 'red', alpha=0.6) 
# ax1.tick_params(axis ='y', labelcolor = 'red') 
  
# # Adding Twin Axes

# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Time taken for execution (s)', color = 'blue') 
# ax2.plot(learning_rates,  mean_total_test[:, 1], color = 'blue', alpha=0.6) 
# ax2.tick_params(axis ='y', labelcolor = 'blue') 
# fig.tight_layout()
# plt.show()


########## Epochs Number

# epochs_rate = [x for x in range(1000, 16000, 3000)]
# test1 = [46.67, 46.67, 100.0, 100.0, 100.0]
# test2 = [26.67, 56.67, 56.67, 56.67, 56.67]
# test3 = [100.0, 100.0, 96.67, 96.67, 96.67]
# test_total = np.add(test1, test2)
# total = np.add(test_total, test3) / 3
# total_ar = []
# for item in total:
#     total_ar.append(item)
# print(total_ar)
# print(len(total_ar), len(epochs_rate))
# plt.bar(epochs_rate, total_ar, color="b", alpha=0.6,width = 200)
# plt.ylim([50,100])
# plt.xlabel("Number of epochs")
# plt.ylabel("Accuracy (%)")
# plt.title("Number of epochs analysis in terms of results accuracy \n for a learning rate of 0.01 over three trials")
# plt.show()

# tested_output = get_output(nn_trained, validation_train[:5])
# print("Output results:", tested_output)
# print("Actual output:", validation_labels[:5])



########## Function Tests

# function_encode = [0, 1, 2, 3]
# function_name = ["tanh/tanh", "tanh/linear", "sigmoig/leaky_relu", "leaky_relu/sigmoid"]
# epochs_8 = [[100.0, 51.11, 100.0, 35.56], [100.0, 15.56, 100.0, 35.56], [15.56, 100.0, 100.0, 57.78]]
# test1 = [97.78, 100.0, 100.0, 40.0]
# test2 = [97.78, 97.78, 97.78, 35.56]
# test3 = [93.33, 100.0, 100.0, 33.33]
# test1, test2, test3 = epochs_8[0], epochs_8[1], epochs_8 [2]
# test_total = np.add(test1, test2)
# total = np.add(test_total, test3) / 3
# total_ar = []
# for item in total:
#     total_ar.append(item)
# plt.bar(function_encode, total_ar, color="b", alpha=0.6)
# plt.xticks(function_encode, function_name)
# plt.xlabel("Type of functions (hidden/output)")
# plt.ylabel("Accuracy (%)")
# plt.title("Function analysis for hidden and out layers in terms of results accuracy \n for a learning rate of 0.01 and epochs of 7,000 over three trials")
# plt.show()

# tested_output = get_output(nn_trained, validation_train[:5])
# print("Output results:", tested_output)
# print("Actual output:", validation_labels[:5])