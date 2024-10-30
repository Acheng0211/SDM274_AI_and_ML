import os
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt

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