import numpy as np
import matplotlib.pyplot as plt
import random

class MLP(object):
    def __init__(self, h_nodes, o_nodes, x_data, y_data):
        self.nb_hidden = h_nodes
        self.nb_output = o_nodes
        self.who = np.array([random.random(), random.random(), random.random()])
        self.wih = np.array([random.random(), random.random(), random.random()])
        self.learning_rate = 0.01
        self.bias = 0.01
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
        N = np.exp(input) - np.exp(-input)
        D =  np.exp(input) + np.exp(-input)
        return N/D
    
    def train(self):
        for _ in range(self.epochs):
            for j in range(len(self.dataset)):
                inputs_hidden = np.dot(self.dataset[j],self.wih) + self.bias
                hidden_activation = self.sigmoid(inputs_hidden)
                hidden_outputs = np.dot(hidden_activation, self.who)
                output = self.tanh(hidden_outputs)
                output_errors = (self.labels[j] - output)**2
                hidden_errors = np.dot(self.who.T, output_errors)
                self.who += self.learning_rate * np.dot(output_errors * (1 - output**2), hidden_outputs)
                self.wih += self.learning_rate * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), self.dataset[j])
        pass
    
    def query(self, input):
        input_hidden = np.dot(input, self.wih)
        hidden_activation = self.sigmoid(input_hidden)
        output = self.tanh(np.dot(hidden_activation, self.who))
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
