import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(file_name):
    # 读取数据集
    column_names = ['class_label'] + [f'feature_{i}' for i in range(13)]  
    wine_data = pd.read_csv(file_name, header=None, names=column_names)

    # 显示原始数据集的类别分布
    # print("Original dataset class distribution:")
    # print(wine_data['class_label'].value_counts())

    return wine_data

def filter(data):
    # 删除 class_label = 3 的所有行
    wine_data_filtered = data[data['class_label'] != 3]

    # 显示新数据集的类别分布
    # print("\nNew dataset class distribution:")
    # print(wine_data_filtered['class_label'].value_counts())

    # # 检查新数据集的形状
    # print("\nOriginal dataset shape:", data.shape)
    # print("New dataset shape:", wine_data_filtered.shape)

    return wine_data_filtered

def classify_data(data):
    # 将数据分为特征和标签
    X = data.drop('class_label', axis=1).values #wine_data_filtered
    y = data['class_label'].values

    return X,y

def split_data(X, y, test_size=0.3, random_state = 42):
    # 划分训练集和测试集，这里我们按照70%训练集，30%测试集的比例来划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    y_train[y_train == 2] = -1
    y_test[y_test == 2] = -1
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    # 打印结果以确认划分
    # print("Training set shape:", X_train.shape)
    # print("Test set shape:", X_test.shape)

    return X_train, X_test, y_train.reshape(-1,1), y_test.reshape(-1,1)