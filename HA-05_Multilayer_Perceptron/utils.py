import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

def generate_data():
    # 生成数据集
    np.random.seed(42)
    # 定义数据集的大小
    dataset_size = 10
    # 生成随机的x和y坐标
    x = np.random.uniform(-5, 5, dataset_size)
    y = np.random.uniform(-5, 5, dataset_size)
    # 生成颜色标签:红色和绿色
    colors = np.random.choice(['red', 'green'], dataset_size)

    # 创建一个DataFrame来保存数据集
    data = pd.DataFrame({
        'color': colors,
        'x': x,
        'y': y,
    })
    # 显示数据集的前几行
    print(data.head())
    save_path = './data'
    # 保存数据集为CSV文件（如果需要）
    data.to_csv(save_path + '/' + 'color_dataset.csv', index=False)

def generate_dataset():
    np.random.seed(3407)
    class_1 = np.hstack([np.random.normal( 1, 1, size=(500, 2)),  np.ones(shape=(500, 1))])
    class_2 = np.hstack([np.random.normal(-1, 1, size=(500, 2)), -np.ones(shape=(500, 1))])
    dataset = np.vstack([class_1, class_2])
    X, y = dataset[:,:2], dataset[:,2]
    return X, y


def load_data(file_name):
    # 读取数据集
    column_names = ['class_label'] + [f'feature_{i}' for i in range(13)]  
    data_raw = pd.read_csv(file_name, header=None, names=column_names)

    # 显示原始数据集的类别分布
    # print("Original dataset class distribution:")
    # print(data_raw['class_label'].value_counts())

    return data_raw

def filter(data):
    # 删除 class_label = 3 的所有行
    data_filtered = data[data['class_label'] != 3]

    # 显示新数据集的类别分布
    # print("\nNew dataset class distribution:")
    # print(data_filtered['class_label'].value_counts())

    # # 检查新数据集的形状
    # print("\nOriginal dataset shape:", data.shape)
    # print("New dataset shape:", data_filtered.shape)

    return data_filtered

def classify_data(data):
    # 将数据分为特征和标签
    X = data.drop('class_label', axis=1).values #data_filtered
    y = data['class_label'].values

    return X, y

def split_data(X, y, test_size=0.3, val_size = 0.2, random_state = 42):
    # 划分训练集和测试集，这里我们按照70%训练集，30%测试集的比例来划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    #Perceptron时-1, LogisticRegression时0
    # y_train[y_train == 2] = -1
    # y_test[y_test == 2] = -1
    # y_train[y_train == 2] = 0
    # y_test[y_test == 2] = 0
    # y_train = y_train.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)

    # 打印结果以确认划分
    # print("Training set shape:", X_train.shape)
    # print("Test set shape:", X_test.shape)

    return X_train, X_val, X_test, y_train.reshape(-1,1), y_val.reshape(-1,1),y_test.reshape(-1,1)


# if __name__ == '__main__':
#     X,y= generate_dataset()
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
#     plt.show()
