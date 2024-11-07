import hydra
import hydra.conf
import wandb
from omegaconf import DictConfig
import utils
from model import MLP
import os
import matplotlib.pyplot as plt


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
    