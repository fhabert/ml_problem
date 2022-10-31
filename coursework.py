from ast import Num
import numpy as np
import matplotlib.pyplot as plt
import random

class MLP(object):
    def __init__(self, h_nodes, o_nodes, x_data, y_data):
        self.nb_hidden = h_nodes
        self.nb_output = o_nodes
        self.who = np.array([0.1, 0.1, 0.1])
        self.wih = np.array([0.1, 0.1, 0.1])
        self.learning_rate = 0.01
        self.dataset = x_data
        self.labels = y_data
        self.epochs = 2000
        pass
    
    def sigmoid(self, input):
        if type(input) == list:
            results = []
            for item in input:
                results.append(1/(1+np.exp(-item)))
            return results
        else:
            return 1/(1+np.exp(-input))
        
    def tanh(self, input):
        N = np.exp(input) - np.exp(-input)
        D =  np.exp(input) + np.exp(-input)
        return N/D
    
    def train(self):
        for _ in range(self.epochs):
            for j in range(len(self.dataset)):
                inputs_hidden = self.dataset[j] * self.wih
                hidden_activation = self.sigmoid(inputs_hidden)
                hidden_outputs = hidden_activation * self.who
                output = self.tanh(sum(hidden_outputs))
                deltas = []
                for i in range(len(self.who)):
                    diff_tan = 1 - (output ** 2)
                    delta = (self.labels[j] - output) * diff_tan
                    deltas.append(delta)
                    self.who[i] += self.learning_rate * delta * hidden_activation[i]
                for i in range(len(self.wih)):
                    sum_delta = np.sum(np.array([deltas[k] * self.who[k] for k in range(len(self.who))]))
                    diff_sig = (1 - hidden_activation[i]) * hidden_activation[i]
                    self.wih[i] += self.learning_rate * sum_delta * self.dataset[j] * diff_sig
        pass
    
    def query(self, input):
        input_hidden = input * self.wih
        hidden_activation = self.sigmoid(input_hidden)
        output = self.tanh(sum(hidden_activation * self.who))
        return output
    
    def visualize_result(self, x, y):
        plt.scatter(self.dataset, self.labels, marker="x", color="r")
        plt.plot(x, y, alpha=0.7, color="b")
        plt.show()
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
mlp = MLP(n_hidden, n_output, x, y)
mlp.train()

for item in test_inputs:
    query_data = mlp.query(item)
    test_y.append(query_data)
    
mlp.visualize_result(test_inputs, test_y)
