import numpy as np
import matplotlib.pyplot as plt
import math
import random

class MLP(object):
    def __init__(self, h_nodes, o_nodes, x_data, y_data):
        self.nb_hidden = h_nodes
        self.nb_output = o_nodes
        self.who = np.array([random.random(), random.random(), random.random()])
        self.wih = np.array([random.random(), random.random(), random.random()])
        self.bias = np.array([random.random(), random.random(), random.random()])
        self.learning_rate = 0.008
        self.dataset = x_data
        self.labels = y_data
        self.epochs = 6000
        pass
    
    def sigmoid(self, input):
        results = []
        for item in input:
            results.append(1/(1 + np.exp(-item)))
        return results
    
    def leaky_relu(self, input):
        if type(input) == np.ndarray:
            results = []
            for item in input:
                if item > 0:
                    results.append(item)
                else:
                    results.append(0.01*item)
            return results
        else:
            if input > 0:
                return input
            return 0
        
        
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
                inputs_hidden = np.dot(self.dataset[j],self.wih) + self.bias
                if func == 1:
                    hidden_activation = self.tanh(inputs_hidden)
                elif func == 2:
                    hidden_activation = self.leaky_relu(inputs_hidden)
                hidden_outputs = np.dot(hidden_activation, self.who)
                output = 1*hidden_outputs
                output_errors = (self.labels[j] - output)
                who_old = self.who
                self.who += self.learning_rate * np.dot(output_errors, hidden_activation)
                for i in range(len(self.wih)):
                    delta = who_old[i] * 1/2 * (output_errors)
                    if func == 1:
                        self.bias[i] += self.learning_rate * delta * (1-hidden_activation[i]**2)
                        self.wih[i] += self.learning_rate * delta * self.dataset[j] * (1-hidden_activation[i]**2)
                    elif func == 2:
                        derivative_relu = 0.01 if hidden_activation[i] < 0 else 1
                        self.bias[i] += self.learning_rate * delta * derivative_relu
                        self.wih[i] += self.learning_rate * delta * self.dataset[j] * derivative_relu 
        pass
    
    def query(self, input):
        input_hidden = np.dot(input, self.wih)
        hidden_activation = self.tanh(input_hidden)
        output = np.dot(hidden_activation, self.who)
        return output
    
    def visualize_result(self, x, y, func):
        if func == 1:
            name = "tanh"
        plt.scatter(self.dataset, self.labels, marker="x", color="r")
        plt.plot(x, y, alpha=0.7, color="b")
        plt.title(f"Non-linear regression neural network based on stochastic \n gradient descent using {name} for the hidden layer")
        plt.show()
        # plt.savefig(f"./problem1_{name}.png")
        pass

x = np.arange(-1, 1, 0.05)
y = []
norm_noise = np.random.normal(0,0.02,len(x))
for item in x:
    rand_index = random.randint(0, len(x)-1)
    result = (item**3) * 0.8 + 0.3 * (item ** 2) - 0.4 * item + norm_noise[rand_index]
    y.append(result)

n_hidden = 3
n_output = 1
test_inputs = np.arange(-0.97, 0.93, 0.1)
test_y = []
tanh, leaky_relu, sigmoid = 1, 2, 3

mlp = MLP(n_hidden, n_output, x, y)
mlp.train(tanh)

for item in test_inputs:
    query_data = mlp.query(item)
    test_y.append(query_data)

mlp.visualize_result(test_inputs, test_y, tanh)
