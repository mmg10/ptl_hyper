from optuna import trial
from scipy.sparse import data
from sklearn.utils import shuffle
import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import DataLoader

import utils

import pytorch_lightning as pl
import optuna

DEVICE = "cpu"
EPOCHS = 1

def run_training(fold, params):
    df = pd.read_csv("./input/train_features.csv")
    df = df.drop(['cp_time', 'cp_dose', 'cp_type'], axis=1)

    targets_df = pd.read_csv("./input/train_targets_fold.csv")

    features = df.drop('sig_id', axis=1).columns
    targets = targets_df.drop(['sig_id', 'kfold'], axis=1).columns

    df = df.merge(targets_df, on='sig_id', how='left')
    # print(df.info())

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[features].to_numpy()
    ytrain = train_df[targets].to_numpy()

    xvalid = valid_df[features].to_numpy()
    yvalid = valid_df[targets].to_numpy()

    train_dataset = utils.MoADataset(features=xtrain, targets=ytrain)
    valid_dataset = utils.MoADataset(features=xvalid, targets=yvalid)

    train_loader = torch.utils.data.DataLoader(
                                            train_dataset,
                                            batch_size = 256,
                                            num_workers = 4,
                                            shuffle=True
                                            )

    valid_loader = torch.utils.data.DataLoader(
                                            valid_dataset,
                                            batch_size = 256,
                                            num_workers = 4,
                                            shuffle=False
                                            )
    mymodel = utils.Model(nfeatures=xtrain.shape[1],
                        ntargets=ytrain.shape[1],
                        nlayers=params['num_layers'],
                        hidden_size=params['hidden_size'],
                        dropout=params['dropout'])

    plmodel = utils.Model_TL(mymodel, fold, params['learning_rate'])
    trainer = pl.Trainer(max_epochs=EPOCHS, num_sanity_val_steps=0, gpus=0, log_every_n_steps=1, enable_model_summary=False)
    trainer.logger.log_hyperparams(params)
    trainer.fit(plmodel, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    return trainer.callback_metrics["val_loss"].item()

def objective(trial):
    params = {
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "hidden_size": trial.suggest_int("hidden_size", 16, 100),
                "dropout": trial.suggest_uniform("dropout", 0.1, 0.7),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-3),
            }
    all_losses = []
    for f_ in range(5):
        temp_loss = run_training(f_, params)
        all_losses.append(temp_loss)

    return np.mean(all_losses)

if __name__ == "__main__":
    # run_training(fold = 0)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    print('best trial:')
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)

    # This code block trains the model using the best parameters found 
    scores = 0
    for j in range(5):
        score = run_training(j, trial_.params)
        scores += score

    print(scores / 5)

