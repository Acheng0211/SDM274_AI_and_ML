import os
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
#https://zhuanlan.zhihu.com/p/501743440
class LinearLayer:
    def __init__(self, n_in, n_out, batch_size, n_feature, n_unit, activation, lr):
        self.W = np.random.randn(n_in, n_out) * 0.01
        self.b = np.zeros((batch_size, n_out))
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.params = {'name': 'Linear', 'size':[n_in, n_out],'W': self.W, 'b': self.b, 'activation': self.activation, 'lr': self.lr}

    def forward_propagation(self, x):
        self.x = x
        output = np.dot(x, self.W) + self.b
        if self.activation is 'sigmoid':
            output = 1 / (1 + np.exp(-output))
        self.activated_output = output
        return output
    
    def backward_propagration(self, dout):
        if self.activation is 'sigmoid':
            dout = self.activated_output * (1 - self.activated_output) * dout
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = dout
        self.W = self.W - self.dW * self.lr / self.batch_size
        self.b = self.b - self.db * self.lr / self.batch_size
        return dx

class Sigmoid:
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.params = {'name': 'Sigmoid'}

    def forward_propagation(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))
    
    def backward_propagation(self, dout):
        return self.x * (1 - self.x) * dout

class MultyLayerPerceptron:
    def __init__(self, input_size, batch_size, num_classes, gd, wandb, lr=0.001, hidden_layer_sizes=(256,), activation='sigmoid'):
        self.wandb = wandb
        self.sigmoid = Sigmoid()
        self.batch_size = batch_size
        self.lr = lr
        self.gd = gd
        self.activation = activation
        #线性层列表初始化
        self.layer_list = [[hidden_layer_sizes[i], hidden_layer_sizes[i + 1]]
                           for i in range(len(hidden_layer_sizes) - 1)]
        self.input_layer = LinearLayer(input_size, hidden_layer_sizes[0], batch_size, activation, lr=lr)
        self.classifier = LinearLayer(hidden_layer_sizes[-1], num_classes, batch_size, activation, lr=lr)
        #将输入层、隐藏层、输出层、激活函数层组合成一个list
        self.layers = [self.input_layer]
        for i in range(len(self.layer_list)):
            self.layers.append(LinearLayer(self.layer_list[i][0], self.layer_list[i][1], batch_size, activation, lr=lr))
        self.layers.append(self.classifier)
        self.layers.append(self.sigmoid)
    
    def forward_propagation(self, x):
        for layer in self.layers:
            x = layer.forward_propagation(x)
        return x
    
    def backward_propagation(self, dout):
        for layer in reversed(self.layers): #比self.layers[::-1]更快,节省内存，无需切片返回一个新列表
            dout = layer.backward_propagation(dout)
        return dout

    def train(self, X, y): 
        break_out = False
        epoch_no_improve = 0

        X = self._preprocess_data(X)
        for epoch in range(self.epoch):
            y_pred = self._predict_probablity(X)
            loss = self._loss(y, y_pred)
            self.loss.append(loss)
            loss_error = np.abs(loss - self.best_loss)
            if self.wandb:
                wandb.log({"loss": loss})
            if loss < self.best_loss - self.tol:
                self.best_loss = loss
                epoch_no_improve = 0
            elif loss_error < self.tol:
                epoch_no_improve += 1
                if epoch_no_improve >= self.patience:
                    # print(f'loss error is {loss_error}, smaller than tolerance, quit at epoch {epoch}')
                    print(f"Early stopping triggered at {epoch} due to the no improvement in loss")
                    break_out = True
                    break
                else:
                    epoch_no_improve = 0
            
            if self.gd == 'SGD':
                i = np.random.randint(0, len(X))
                grad = self._gradient(np.expand_dims(X[i], axis=0), np.expand_dims(y[i], axis=0), np.expand_dims(y_pred[i], axis=0))
            elif self.gd == 'MBGD':
                batch_size = 10
                idx = np.random.choice(y.shape[0], batch_size, replace=False)
                grad = self._gradient(X[idx], y[idx], y_pred[idx]) 
            self.W -= self.lr * grad 

            if break_out:
                break_out = False
                break

    def _linear_tf(self, X):
        return X @ self.W
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z):
        sigma = self._sigmoid(z)
        return sigma * (1 - sigma)

    def _mse_loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    def _cross_entropy_loss(self, y, y_pred):
        epsilon = 0
        return -np.mean(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon))

    def _gradient(self, X, y, y_pred):
        return (-(y - y_pred).T @ X / y.size).reshape(-1, 1)
        # return - 1 / y_pred.shape[0] * X.T @ (y - y_pred).reshape(-1, 1)

    def _preprocess_data(self,X):
        m, n = X.shape
        X_ = np.empty([m, n+1])
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def _predict_probablity(self, X): 
        z = self._linear_tf(X)
        prob = self._sigmoid(z)
        data = np.concatenate((z, prob), axis=1)
        # np.savetxt("compare.csv", data, delimiter=", ")
        return prob

    def _predict(self, X): 
        X = self._preprocess_data(X)
        y_pred = self._predict_probablity(X)
        return np.where(y_pred>=0.5,1,0)

    def evaluate(self, X_test, y_test):
        TP,FP,FN,TN = 0,0,0,0
        X_test = self._predict(X_test)
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