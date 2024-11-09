import hydra
import hydra.conf
import wandb
from omegaconf import DictConfig
import utils as uts
from model import kNN
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


@hydra.main(version_base="1.3", config_path="./conf", config_name="config_knn")
def main(cfg: DictConfig):

    # Preprocess dataset
    dataset_path = cfg.dataset_path
    print("If path is existed:", os.path.exists(dataset_path))
    data_raw = uts.load_data(dataset_path)
    X, y = uts.classify_data(data_raw)
    X_train, X_test, y_train, y_test = uts.split_data(X, y, test_size=0.3, random_state=42)
    
    if(cfg.wandb_on_off):
        wandb.init(project="knn")

    k_values = range(1, cfg.k_max+1)
    accuracies = []

    for k in k_values:
        knn = kNN(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f'k={k}, Accuracy={accuracy:.4f}')

    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs k in kNN')
    plt.savefig(os.path.join('./output','Accvsk.png'))
    plt.show()

if __name__ == "__main__":
    main()
