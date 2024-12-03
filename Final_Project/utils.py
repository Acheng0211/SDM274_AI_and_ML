import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter='\s+', header=None) # delimiter='\s+' for space separated values
    X = data.iloc[:, :-1].values 
    y = data.iloc[:, -1].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def reconstruction_error(X, X_reconstructed):
    return np.mean((X[:,:X_reconstructed.shape[1]] - X_reconstructed) ** 2)