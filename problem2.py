import numpy as np
import math
import random
import pandas as pd

dataset = pd.read_csv("./dataset_iris.csv", sep=";", encoding="utf-8", header=None)
df = pd.DataFrame(dataset)

class Classification(object):
    def __init__(self, hidden_layers):
        self.features_train = None
        self.features_labels = None
        self.validation_train = None
        self.validation_labels = None
        self.nb_outputs = None
        self.set_dataframe()
        self.nb_inputs = len(self.features_train[0])
        self.nb_h_layers = hidden_layers
        self.biases = [[random.random() for i in range(x)] for x in self.nb_h_layers] + [[random.random() for _ in range(self.nb_outputs)]]
        self.learning_rate = 0.01
        self.epochs = 1
        self.nodes = self.set_nodes()
        pass
    
    def set_dataframe(self):
        amount_train = round(0.7*len(df))
        titles = df.iloc[:, 0].unique()
        self.nb_outputs = len(titles)
        one_hot_titles = []
        zeroes_string = [0 for _ in range(len(titles))]
        for i in range(len(zeroes_string)):
            category = [0.2 if i != j else 0.8 for j in range(len(zeroes_string))]
            one_hot_titles.append(category)
        l = len(df)
        df_randomized = [None for _ in range(l)]
        for i in range(l):
            rand = random.randint(0, l-1)
            if type(df_randomized[rand]) == None:
                df_randomized[rand] = list(df.iloc[i])
            else:
                while rand <= l and df_randomized[rand] != None:
                    if rand >= l-1:
                        rand = 0
                    else:
                        rand += 1
                df_randomized[rand] = list(df.iloc[i])
        
        for i in range(len(df_randomized)):
            if df_randomized[i][0] == "Iris-virginica":
                df_randomized[i][0] = one_hot_titles[0]
            elif df_randomized[i][0] == "Iris-setosa":
                df_randomized[i][0] = one_hot_titles[1]
            else:
                df_randomized[i][0] = one_hot_titles[2]
        new_df = np.array(df_randomized,dtype=object)
        self.features_train = new_df[:, 1:][:amount_train]
        self.features_labels = new_df[:, 0][:amount_train]
        self.validation_train = new_df[:, 1:][amount_train:]
        self.validation_labels = new_df[:, 0][amount_train:]


    def tanh(self, input):
        if type(input) == np.ndarray or type(input) == list:
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
        
    def sigmoid(self, input_data):
        if type(input_data) == np.ndarray:
            results = []
            for item in input_data:
                results.append(1/(1 + np.exp(-item)))
            return results
        else:
            return 1/(1 + np.exp(-input_data))

    def set_nodes(self):
        nodes = {"inputs": [], "hidden": [], "outputs": [] }
        for i in range(self.nb_inputs):
            weights = [random.random() for _ in range(self.nb_h_layers[0])]
            n_inputs = Node(weights, f"i{i}")
            nodes["inputs"].append(n_inputs)
        all_layers = []
        for i in range(len(self.nb_h_layers)):
            all_layers.append(self.nb_h_layers[i])
        all_layers.append(3)
        for i in range(len(self.nb_h_layers)):
            nodes_hidden = []
            nb = self.nb_h_layers[i]
            for k in range(nb):
                weights = [random.random() for _ in range(all_layers[i+1])]
                n_hidden = Node(weights, f"h{i}{k}", self.biases[i][k])
                nodes_hidden.append(n_hidden)
            nodes["hidden"].append(nodes_hidden)
        for i in range(self.nb_outputs):
            n_output = Node(None, f"o{i}", self.biases[-1][i])
            nodes["outputs"].append(n_output)
        return nodes

    def forward_propag(self, input_feature):
        layers = []
        ordered_weights = np.array([self.nodes["inputs"][l].w for l in range(len(self.nodes["inputs"]))])
        hidden_from_inputs = []
        counter_hidden_layers = 0
        # Input to hidden layer
        for k in range(len(self.nodes["inputs"][0].w)):
            weights = ordered_weights[:, k]
            sum_sigma = 0
            for l in range(len(self.nodes["inputs"])):
                input_value = float(input_feature[l])
                sum_sigma += input_value * weights[l]
            sum_sigma += self.nodes["hidden"][0][k].bias
            h_active = self.tanh(sum_sigma)
            hidden_from_inputs.append(h_active)
        LayerOne = Layer(hidden_from_inputs, f"L{counter_hidden_layers}")
        layers.append(LayerOne)

        # Hidden to hidden and hidden to output
        for k in range(len(self.nb_h_layers)-1):
            if k == 0:
                last_hidden_activation = layers[0].hidden_activation
            else:
                last_hidden_activation = layers[-1].hidden_activation
            last_hidden_nodes = self.nodes["hidden"][k]
            hidden_layer_activation = []
            sum_sigma = 0
            ordered_weights = np.array([last_hidden_nodes[l].w for l in range(len(last_hidden_nodes))])
            counter_hidden_layers += 1
            for m in range(len(last_hidden_nodes[0].w)):
                weights = ordered_weights[:, m]
                for j in range(len(last_hidden_nodes)):
                    input_value = last_hidden_activation[j]
                    sum_sigma += input_value * weights[j]
                sum_sigma += last_hidden_nodes[j].bias
                h_active = self.tanh(sum_sigma)
                hidden_layer_activation.append(h_active)
            LayerHidden = Layer(hidden_layer_activation, f"L{counter_hidden_layers}")
            layers.append(LayerHidden)

        # Output
        last_hidden = layers[-1].hidden_activation
        output_hidden = []
        ordered_weights = np.array([self.nodes["hidden"][-1][l].w for l in range(len(self.nodes["outputs"]))])
        for k in range(len(self.nodes["hidden"][-1][0].w)):
            sum_sigma = 0
            weights = ordered_weights[:, k]
            for j in range(len(self.nodes["outputs"])):
                input_value = last_hidden[j]
                sum_sigma += input_value * weights[j]
            sum_sigma += self.nodes["outputs"][j].bias
            h_active = self.tanh(sum_sigma)
            output_hidden.append(h_active)
        counter_hidden_layers += 1
        LayerHidden = Layer(output_hidden, f"L{counter_hidden_layers}")
        layers.append(LayerHidden)
        return layers

    def back_propag(self, layers, d_output):
        output = self.tanh(layers[-1].hidden_activation)
        output_errors = np.sum([(d_output - x) for x in output])
        copy_nodes = {}
        for key, value in self.nodes.items():
            copy_nodes[key] = value
        delta_output = np.dot(output_errors, 1 - np.square(output))
        for i in range(len(self.biases[-1])):
            self.biases[-1][i] += self.learning_rate * delta_output[i]
        
        # Hidden to hidden and hidden to output
        for i in range(len(layers)-1, 0, -1):
            nb_nodes = self.nb_h_layers[i-1]
            for j in range(nb_nodes):
                for k in range(len(self.nodes["hidden"][i-1][j].w)):
                    # print(j)
                    self.nodes["hidden"][i-1][j].w[k] += self.learning_rate * delta_output[k] * layers[i].hidden_activation[k]
        # Input to hidden
        ##############################################################
        print(copy_nodes["hidden"][1][0].w)
        delta_wih = who_old[i] * delta_who * (1-hidden_activation[i]**2)
        self.bias[0][i] += self.learning_rate * delta_wih
        self.wih[i] += self.learning_rate * delta_wih * self.dataset[j]
        for i in range(self.nb_inputs):
            delta_wih = copy_nodes
        pass

    def train(self):
        for _ in range(self.epochs):
            for i in range(1):
                forward_layers = self.forward_propag(self.features_train[i])
                self.back_propag(forward_layers, self.features_labels[i])

        # for item in forward_layers:
        #     print(item.name)
        pass

class Layer(object):
    def __init__(self, hidden_activation, name):
        self.hidden_activation = hidden_activation
        self.name = name
        pass

class Node(object):
    def __init__(self, weights, name, bias=None):
        self.bias = bias
        self.w = weights
        self.name = name
        pass

    # def train(self):
    #     for _ in range(self.epochs):
    #         for j in range(len(self.dataset)):
    #             inputs_hidden = np.dot(self.dataset[j], self.wih)
    #             for i in range(len(inputs_hidden)):
    #                 inputs_hidden[i] += self.bias[0][i]
    #             hidden_activation = self.tanh(inputs_hidden)
    #             hidden_outputs = np.dot(hidden_activation, np.array(self.who)) + self.bias[1][0]
    #             output = self.tanh(hidden_outputs)
    #             output_error = (self.labels[j] - output)
    #             who_old = self.who
    #             delta_who = output_error * (1-output**2)
    #             self.bias[1][0] += self.learning_rate * delta_who
    #             for i in range(len(self.who)):
    #                 self.who[i] += self.learning_rate * delta_who * hidden_activation[i]
    #             for i in range(len(self.wih)):
    #                 delta_wih = who_old[i] * delta_who * (1-hidden_activation[i]**2)
    #                 self.bias[0][i] += self.learning_rate * delta_wih
    #                 self.wih[i] += self.learning_rate * delta_wih * self.dataset[j]
    #     print("who:", self.who)
    #     print("wih:", self.wih)
    #     print("bias:", self.bias)


hidden_layers = [5, 3]
iris = Classification(hidden_layers)
# print(iris.nodes)
iris.train()
# print(iris.wih)


