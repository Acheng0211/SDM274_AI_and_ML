import os
import time
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from typing import Generator, Union, Tuple, Dict
from tqdm import tqdm
#https://github.com/XavierSpycy/NumPyMultilayerPerceptron/blob/main/nn/layers.py

class InputLayer:
    def __init__(self, n_in, n_out, batch_size, n_unit):
        self.V = np.random.randn(n_in, n_out) * 0.01
        self.b_v = np.zeros((batch_size, n_out))
        self.batch_size = batch_size
        self.n_unit = n_unit
        self.params = {'name': 'Input', 'size':[n_in, n_out],'V': self.V, 'b_v': self.b_v, 'n_unit': self.n_unit}

    def forward_propagation(self, x):
        self.x = x
        output = np.dot(x, self.V) + self.b_v
        return output
    
    def backward_propagation(self, dout, output):
        dout = output * (1 - output) * dout
        dx = np.dot(dout, self.V.T)
        self.dV = np.dot(self.x.T, dout)
        self.db_v = dout
        self.V = self.V - self.dV * self.lr / self.batch_size
        self.b_v = self.b_v - self.db_v * self.lr / self.batch_size
        return dx
    
class HiddenLayer:
    def __init__(self, n_in, n_out, batch_size, n_feature, n_unit, activation, lr):
        self.W = np.random.randn(n_in, n_out) * 0.01
        self.b_w = np.zeros((batch_size, n_out))
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.n_unit = n_unit
        self.params = {'name': 'Hidden', 'size':[n_in, n_out],'W': self.W, 'b_w': self.b_w, 'activation': self.activation, 'lr': self.lr}

    def forward_propagation(self, x):
        self.x = x
        output = np.dot(x, self.W) + self.b_w
        if self.activation is 'sigmoid':
            output = 1 / (1 + np.exp(-output))
        self.activated_output = output
        return self.activated_output
    
    def backward_propagation(self, dout):
        if self.activation is 'sigmoid':
            dout = self.activated_output * (1 - self.activated_output) * dout
        d_previous = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db_w = dout
        self.W = self.W - self.dW * self.lr / self.batch_size
        self.b_w = self.b_w - self.db_w * self.lr / self.batch_size
        return d_previous
    
class OutputLayer: #todo: right?
    def __init__(self, n_in, n_out, batch_size, n_unit, activation, lr):
        self.W = np.random.randn(n_in, n_out) * 0.01
        self.b_w = np.zeros((batch_size, n_out))
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.n_unit = n_unit
        self.params = {'name': 'Output', 'size':[n_in, n_out],'W': self.W, 'b_w': self.b_w, 'activation': self.activation, 'lr': self.lr}

    def forward_propagation(self, x):
        self.x = x
        output = np.dot(x, self.W) + self.b_w
        if self.activation is 'sigmoid':
            output = 1 / (1 + np.exp(-output))
        self.activated_output = output
        return output
    
    def backward_propagation(self, dout):
        if self.activation is 'sigmoid':
            dout = self.activated_output * (1 - self.activated_output) * dout
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db_w = dout
        self.W = self.W - self.dW * self.lr / self.batch_size
        self.b_w = self.b_w - self.db_w * self.lr / self.batch_size
        return dx
    
class MLP:
    def __init__(self, input_size, batch_size, num_classes, epoch, gd, wandb, lossf, tol, lr=0.001, hidden_layer_sizes=(256,), activation='sigmoid'):
        self.wandb = wandb
        self.batch_size = batch_size
        self.lr = lr
        self.gd = gd
        self.epoch = epoch
        self.activation = activation
        self.lossf = lossf
        self.loss = []
        self.tol = tol
        #线性层列表初始化
        self.layer_list = [[hidden_layer_sizes[i], hidden_layer_sizes[i + 1]]
                           for i in range(len(hidden_layer_sizes) - 1)]
        self.input_layer = InputLayer(input_size, hidden_layer_sizes[0], batch_size, activation, lr=lr)
        self.hidden_layer = HiddenLayer(hidden_layer_sizes[-1], num_classes, batch_size, activation, lr=lr)
        self.output_layer = OutputLayer(hidden_layer_sizes[-1], num_classes, batch_size, activation, lr=lr)
        
    def _combine_layers(self):  
        #将输入层、隐藏层、输出层、激活函数层组合成一个list
        self.layers = [self.input_layer]
        for i in range(len(self.layer_list)):
            self.layers.append(HiddenLayer(self.layer_list[i][0], self.layer_list[i][1], self.batch_size, self.avscodevscodactivation, lr=lr))
        self.layers.append(self.output_layer)
    
    def forward_propagation(self, x):
        for layer in self.layers:
            if layer.params['name'] == 'Input':
                continue
            x = layer.forward_propagation(x)
        return x
    
    def backward_propagation(self, dout):
        for layer in reversed(self.layers): #比self.layers[::-1]更快,节省内存，无需切片返回一个新列表
            if layer.params['name'] != 'Input':
                dout = layer.backward_propagation(dout)
        return dout
    
    def criterion(self, y, y_pred):
        if self.lossf == 'CrossEntropy':
            y_pred_clip = np.clip(y_pred, 1e-10, 1 - 1e-10)
            loss = -np.sum(y * np.log(y_pred_clip)) / len(y)
            delta = y_pred_clip - y
        return loss, delta

    def fit(self, X_, y_, X_val, y_val, batch_size, progress_bar): 
        break_out = False
        epoch_no_improve = 0
        batch_size = self.batch_size
        X = X_.copy()
        y = y_.copy()
        start_time = time.time()

        X #= self._preprocess_data(X)
        self._combine_layers()
        for epoch in self.conditional_tqdm(range(self.epoch), progress_bar):
            loss_error = np.abs(loss - self.best_loss)
            if self.wandb:
                wandb.log({"loss": loss})
            if self.gd == 'SGD':
                i = np.random.randint(0, len(X))
                grad = self._gradient(np.expand_dims(X[i], axis=0), np.expand_dims(y[i], axis=0), np.expand_dims(y_pred[i], axis=0))
            elif self.gd == 'MBGD':
                idx = np.random.choice(y.shape[0], batch_size, replace=False)
                for i in idx:
                    X_batch = X[i]
                    y_batch = y[i]
                    y_hat = self.forward_propagation(X_batch)
                    loss, delta = self.criterion(y_batch, y_hat)
                    self.backward_propagation(delta)
                    #update the weights
                    i += 1
                    self.loss.append(loss)
            end_time = time.time()

            if loss < self.best_loss - self.tol:
                self.best_loss = loss
                epoch_no_improve = 0
            elif loss_error < self.tol:
                epoch_no_improve += 1
                if epoch_no_improve >= self.patience:
                    y_val_pred = self.predict(X_val)
                    #todo: callback function
                    break_out = callback(y_val, y_val_pred)
                    print(f"Early stopping triggered at {epoch} due to the no improvement in loss")
                    break
                else:
                    epoch_no_improve = 0
            if break_out:
                break_out = False
                break

    def predict(self, X):
        X = np.array(X)
        output = self.forward_propagation(X)
        if self.lossf == 'CrossEntropy':
            # todo:return predicted probability
            return np.argmax(output, axis=1)
        else: 
            return output.reshape(-1)

    def _linear_tf(self, X):
        return X @ self.W
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z):
        sigma = self._sigmoid(z)
        return sigma * (1 - sigma)
    
    def _ReLU(self, z):
        return np.maximum(0, z)
    
    def _tanh(self, z):
        return np.tanh(z)
    
    def _softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def _mse_loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    def _cross_entropy_loss(self, y, y_pred):
        epsilon = 0
        return -np.mean(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon))
    
    def evaluate(self, X_test, y_test):
        TP,FP,FN,TN = 0,0,0,0
        X_test = self.predict(X_test)
        for i in range(len(X_test)):
            y = X_test[i]
            if y == 1 and y_test[i] == 1:
                TP += 1
            elif y == 1 and y_test[i] == -1:
                FP += 1
            elif y == 0 and y_test[i] == 1:
                FN += 1
            elif y == 0 and y_test[i] == -1:
                TN += 1
        # print(TP, FP, FN, TN)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = 2 * precision * recall / (precision + recall)
        print(f"evaluation results:\n    accuracy: {accuracy}, \n    recall: {recall}, \n    precision: {precision}, \n    F1: {F1}")
    
    def plot_loss(self, loss, name):
        plt.plot(loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name}_{self.gd}_loss')
        plt.savefig(os.path.join('./output', f'{name}_{self.gd}_loss.png'))
        plt.show()

    #todo: plot scatter data and classification boundary
    def plot_data(self, X, y, name):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        plt.title(f'{name}_data')
        plt.savefig(os.path.join('./output', f'{name}_data.png'))
        plt.show()

    def conditional_tqdm(self, iterable: range, use_progress_bar: bool=False) -> Generator[int, None, None]:
        if use_progress_bar:
            for item in tqdm(iterable):
                yield item
        else:
            for item in iterable:
                yield item