import hydra
import hydra.conf
import wandb
from omegaconf import DictConfig
import utils
import os
import matplotlib.pyplot as plt
import numpy as np

class HiddenLayer:
    def __init__(self, n_in, n_units, batch_size, activation, lr):
        self.n_in = n_in 
        self.n_out = n_units
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.W = np.random.randn(self.n_in, self.n_out) * 0.01
        self.b = np.zeros(self.n_out)
        self.params = {'name': 'Hidden', 'size':[self.n_in, self.n_out],'W': self.W, 'b': self.b, 'batch_size': self.batch_size,'activation': self.activation, 'lr': self.lr}

    def forward_propagation(self, X):
        self.X = X
        u = np.dot(X, self.W) + self.b
        if self.activation == 'sigmoid':
            h = self.Activation(self.activation, u)
        self.activated_output = h
        return h
    
    def backward_propagation(self, dout):
        if self.activation == 'sigmoid':
            dout = self.Activation_derivative(self.activation, self.activated_output) * dout
        self.dW = np.dot(self.X.T, dout)
        self.db = dout
        self.W -= self.dW * self.lr
        self.b -= self.db * self.lr
    
    def Activation(self, activation, u):
        """
        Return activation function and its derivative
        """
        activation = self.activation
        if activation == 'sigmoid':
            h = 1 / (1 + np.exp(-u))
        return h
    
    def Activation_derivative(self, activation, u):
        """
        Return the derivative of activation function 
        """
        activation = self.activation
        if activation == 'sigmoid':
            d_h = u * (1 - u)
        return d_h

class MLP:
    def __init__(self, input_size, n_units, n_layers, batch_size, loss_function, activation_function, epoch, gd, wandb_on_off, tol, lr):
        """
        
        """
        self.input_size = input_size
        self.n_units = n_units
        self.n_layers = n_layers
        self.wandb = wandb_on_off
        self.batch_size = batch_size
        self.lr = lr
        self.gd = gd
        self.epoch = epoch
        self.activation = activation_function
        self.lossf = loss_function
        self.loss = []
        self.tol = tol
        #线性层列表初始化
        self.layers = []
        self.patience = 100
        self.best_loss = np.inf
        
    def _combine_layers(self):  
        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(HiddenLayer(self.input_size, self.n_units, self.batch_size, self.activation, self.lr))
            # elif i == self.n_layers - 1:
            #     self.layers.append(HiddenLayer(self.layers[i-1].n_out, 1, self.batch_size, self.activation, self.lr))
            else:
                self.layers.append(HiddenLayer(self.layers[i-1].n_out, self.n_units, self.batch_size, self.activation, self.lr))
    
    def forward_propagation(self, x):
        for layer in self.layers:
            x = layer.forward_propagation(x)
        return x
    
    def backward_propagation(self, grad):
        i = self.n_layers
        for layer in reversed(self.layers): #比self.layers[::-1]更快,节省内存，无需切片返回一个新列表
            # print(i)
            i -= 1
            if layer == self.layers[0]:
                break
            dout = layer.backward_propagation(grad) 
        return dout
    
    def fit(self, X_, y_, X_val, y_val): 
        break_out = False
        epoch_no_improve = 0
        batch_size = self.batch_size
        X = X_.copy()
        y = y_.copy()
        loss = 0

        # self._combine_layers()
        for epoch in range(self.epoch):
            loss_error = np.abs(loss - self.best_loss)
            if self.wandb:
                wandb.log({"loss": loss})
            if self.gd == 'SGD':
                i = np.random.randint(0, len(X))
                y_pred = self.forward_propagation(X[i])
                grad = y_pred - y[i]
                loss = self._cross_entropy_loss(y[i], y_pred)
                delta = self.backward_propagation(grad)
                self.loss.append(loss)
            elif self.gd == 'MBGD':
                idx = np.random.choice(X.shape[0], batch_size, replace=False)
                for id in idx:
                    y_pred = self.forward_propagation(X[id])
                    grad = y_pred - y[id]
                    loss = self._cross_entropy_loss(y[i], y_pred)
                    delta = self.backward_propagation(grad)
                    self.loss.append(loss)

            if loss < self.best_loss - self.tol:
                self.best_loss = loss
                epoch_no_improve = 0
            elif loss_error < self.tol:
                epoch_no_improve += 1
                if epoch_no_improve >= self.patience:
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
        return output

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
        print(X_test.shape)
        y_test_pred = self.predict(X_test)
        print(y_test_pred.shape)
        print(y_test.shape)
        data = np.concatenate((y_test, y_test_pred), axis=1)
        np.savetxt("compare.csv", data, delimiter=", ")
        for i in range(len(y_test_pred)):
            if y_test_pred[i] == 1 and y_test[i] == 1:
                TP += 1
            elif y_test_pred[i] == 1 and y_test[i] == -1:
                FP += 1
            elif y_test_pred[i] == 0 and y_test[i] == 1:
                FN += 1
            elif y_test_pred[i] == 0 and y_test[i] == -1:
                TN += 1
        print(TP, FP, FN, TN)
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

@hydra.main(version_base="1.3", config_path="./conf", config_name="config_MLP")
def main(cfg: DictConfig):

    X, y = utils.generate_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data(X, y, test_size=0.3, val_size=0.2, random_state=42)
    
    model = MLP(cfg.input_size, cfg.n_units, cfg.n_layers, cfg.batch_size, cfg.loss_function, cfg.activation_function, cfg.epoch, cfg.gd, cfg.wandb_on_off, cfg.tol, cfg.lr)

    model._combine_layers()
    print(f"num_layers:{len(model.layers)}")
    model.fit(X_train, y_train, X_val, y_val)
    # model.evaluate(X_test, y_test)
    model.plot_loss(model.loss, cfg.name)
    # model.plot_data(X_test, y_test)
    wandb.finish()

if __name__ == "__main__":
    main()