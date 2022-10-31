import numpy as np
import matplotlib.pyplot as plt
import random

class MLP(object):
    def __init__(self, h_nodes, o_nodes, x_data, y_data):
        self.nb_hidden = h_nodes
        self.nb_output = o_nodes
        self.who = np.array([0.5, 0.5, 0.5])
        self.wih = np.array([0.5, 0.5, 0.5])
        self.learning_rate = 0.01
        self.dataset = x_data
        self.labels = y_data
        self.epochs = 1500
        self.w0 = 0
        self.w1 = 0
        self.w2 = 0
        pass
    
    def activation_func(self, input):
        if type(input) == np.ndarray:
            results = []
            for item in input:
                if item > 0:
                    results.append(item)
                else:
                    results.append(0.01)
            return results
        else:
            if input > 0:
                return input
            return 0.01 
    
    def train(self):
        for _ in range(self.epochs):
            w0_test = 0
            w1_test = 0
            w2_test = 0
            for j in range(len(self.dataset)):
                inputs_hidden = np.dot(self.dataset[j], self.wih)
                hidden_activation = self.activation_func(inputs_hidden)
                hidden_outputs = np.dot(hidden_activation, self.who)
                output = self.activation_func(hidden_outputs)
                deltas = []
                for i in range(len(self.who)):
                    derivative_relu = 1 if output > 0 else 0.01
                    delta = (self.labels[j] - output) * derivative_relu
                    deltas.append(delta)
                    self.who[i] += self.learning_rate * delta * hidden_activation[i]
                for i in range(len(self.wih)):
                    sum_delta = np.sum(np.array([deltas[j] * self.who[j] for j in range(len(self.who))]))
                    derivative_relu = 1 if hidden_activation[i] > 0 else 0.01
                    self.wih[i] += self.learning_rate * sum_delta * self.dataset[i]
                # w0_test += - 2*((self.labels[j] - self.w0*(self.dataset[j]**3) - self.w1*(self.dataset[j]**2) - self.w2*self.dataset[j])) \
                #             * (self.dataset[j])
                # w1_test += - 2*((self.labels[j] - self.w0*(self.dataset[j]**3) - self.w1*(self.dataset[j]**2) - self.w2*self.dataset[j])) \
                #             * (2 * self.dataset[j])
                # w2_test += - 2*((self.labels[j] - self.w0*(self.dataset[j]**3) - self.w1*(self.dataset[j]**2) - self.w2*self.dataset[j])) \
                #             * (3 * self.dataset[j]**2)
            # self.w0 -= self.learning_rate*w0_test
            # self.w1 -= self.learning_rate*w1_test
            # self.w2 -= self.learning_rate*w2_test
        pass
    
    def query(self, input):
        input_hidden = np.dot(input, self.wih)
        hidden_activation = self.activation_func(input_hidden)
        output = np.dot(hidden_activation, self.who)
        return output
        # output = self.w0 * (input ** 3) + self.w1 * (input ** 2) + self.w2 * input
        # return output
    
    def visualize_result(self, x, y):
        plt.scatter(self.dataset, self.labels, marker="x", color="r")
        plt.plot(x, y, alpha=0.7, color="b")
        plt.show()
        
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