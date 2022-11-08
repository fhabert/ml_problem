import numpy as np
import matplotlib.pyplot as plt
import math
import random

class MLP(object):
    def __init__(self, h_nodes, o_nodes, x_data, y_data):
        self.nb_hidden = h_nodes
        self.nb_output = o_nodes
        self.who = [random.random() for _ in range(h_nodes)]
        self.wih = [random.random() for _ in range(h_nodes)]
        self.bias = [[random.random() for _ in range(h_nodes)]] + [[random.random()]]
        self.learning_rate = 0.007
        self.dataset = x_data
        self.labels = y_data
        self.epochs = 5000
        pass
    
    def sigmoid(self, input):
        results = []
        for item in input:
            results.append(1/(1 + np.exp(-item)))
        return results
    
    def leaky_relu(self, input):
        results = []
        for item in input:
            if item > 0:
                results.append(item)
            else:
                results.append(0.01)
        return results
        
    def tanh(self, input):
        if type(input) == np.ndarray:
            results = []
            for item in input:
                N = np.exp(item) - np.exp(-item)
                D =  np.exp(item) + np.exp(-item)
                results.append(N/D)
            return results
        else:
            N = np.exp(input) - np.exp(-input)
            D =  np.exp(input) + np.exp(-input)
            return N/D
    
    def train(self, func):
        for _ in range(self.epochs):
            for j in range(len(self.dataset)):
                inputs_hidden = np.dot(self.dataset[j], self.wih)
                for i in range(len(inputs_hidden)):
                    inputs_hidden[i] += self.bias[0][i]
                if func[0] == 1:
                    hidden_activation = self.tanh(inputs_hidden)
                elif func[0] == 2:
                    hidden_activation = self.leaky_relu(inputs_hidden)
                hidden_outputs = np.dot(hidden_activation, np.array(self.who)) + self.bias[1][0]
                if func[1] == 1:
                    output = self.tanh(hidden_outputs)
                elif func[1] == 3:
                    output = hidden_outputs
                output_error = (self.labels[j] - output)
                who_old = self.who
                if func[1] == 1:
                    delta_who = output_error * (1-output**2)
                elif func[1] == 3:
                    delta_who = output_error * 1
                self.bias[1][0] += self.learning_rate * delta_who
                for i in range(len(self.who)):
                    self.who[i] += self.learning_rate * delta_who * hidden_activation[i]
                for i in range(len(self.wih)):
                    if func[0] == 1:
                        delta_wih = who_old[i] * delta_who * (1-hidden_activation[i]**2)
                    elif func[0] == 2:
                        delta_wih = who_old[i] * delta_who * (1-hidden_activation[i])*hidden_activation[i]
                    self.bias[0][i] += self.learning_rate * delta_wih
                    self.wih[i] += self.learning_rate * delta_wih * self.dataset[j]
        print("who:", self.who)
        print("wih:", self.wih)
        print("bias:", self.bias)
        pass
    
    def query(self, input):
        inputs_hidden = np.dot(input, self.wih)
        for i in range(len(inputs_hidden)):
            inputs_hidden[i] += self.bias[0][i]
        hidden_activation = self.tanh(inputs_hidden)
        hidden_outputs = np.dot(hidden_activation, np.array(self.who)) + self.bias[1][0]
        output = output = self.tanh(hidden_outputs)
        return output
    
    def visualize_result(self, x, y):
        plt.scatter(self.dataset, self.labels, marker="x", color="r")
        plt.plot(x, y, alpha=0.7, color="b")
        plt.title("Non linear regression based on tanh activation function")
        plt.show()
        pass
    

x = np.arange(-1, 1, 0.05)
y = []
norm_noise = np.random.normal(0,0.02,len(x))
for item in x:
    rand_index = random.randint(0, len(x)-1)
    result = (item**3) * 0.8 + 0.3 * (item ** 2) - 0.4 * item + norm_noise[rand_index]
    y.append(result)

n_hidden = 5
n_output = 1
test_inputs = np.arange(-0.97, .93, 0.1)
test_y = []
tanh, leaky_relu, linear = 1, 2, 3
mlp = MLP(n_hidden, n_output, x, y)
test_activation_functions = [[tanh, linear], [leaky_relu, linear], [tanh, tanh]]
test_combinaison = 2

mlp.train(test_activation_functions[test_combinaison])

for item in test_inputs:
    query_data = mlp.query(item)
    test_y.append(query_data)

mlp.visualize_result(test_inputs, test_y)