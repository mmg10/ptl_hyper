import torch
import torch.nn as nn

import time
import datetime

import pytorch_lightning as pl
import tableprint as tp

from train import run_training


class MoADataset:
    
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, item):
        return {
            "x": torch.tensor(self.features[item, :], dtype=torch.float),
            "y": torch.tensor(self.targets[item, :], dtype=torch.float),
        }

class Model(nn.Module):
    
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size, dropout):
        super().__init__()
        layers = []

        for _ in range(nlayers):
            if len(layers) == 0:
                layers.append(nn.Linear(nfeatures, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, ntargets))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):   
        output = self.model(x)
        return output

class Model_TL(pl.LightningModule):
    def __init__(self, model, fold, learning_rate):
        super(Model_TL, self).__init__()
        self.model = model
        self.fold = fold
        self.avg_train_loss = 0.
        self.avg_valid_loss = 0.
        self.avg_test_loss = 0.
        self.table_context = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.start_time = 0
        self.end_time = 0
        self.epoch_mins = 0
        self.epoch_secs = 0
        self.table_context = None
        # self.save_hyperparameters("learning_rate")
        self.learning_rate = learning_rate
        

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optim


    def training_step(self, batch, batch_idx):
        inputs, targets = batch['x'], batch['y']
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return {"loss": loss, "p": outputs.detach(), "y": targets}
    
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch['x'], batch['y']
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return {"loss": loss, "p": outputs.detach(), "y": targets}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch['x'], batch['y']
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return {"loss": loss, "p": outputs.detach(), "y": targets}


    def on_train_epoch_start(self) :
        self.start_time = time.time()

    def test_epoch_end(self, outputs):        
        self.avg_test_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        

        self.log("epoch_num", (self.current_epoch+1.0), on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.log("test_loss", self.avg_test_loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)

    def validation_epoch_end(self, outputs):
        if self.trainer.sanity_checking:
            return

        self.avg_valid_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.log("epoch_num", (self.current_epoch+1.0), on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.log("val_loss", self.avg_valid_loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
                 

    def training_epoch_end(self, outputs):
        self.avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean().item()

    def on_train_epoch_end(self):
        self.end_time = time.time()
        time_int = self.format_time(self.start_time, self.end_time)
    
        metrics = {'epoch': self.current_epoch+1, 'Fold': self.fold, 'Train Loss': self.avg_train_loss,  'Valid Loss': self.avg_valid_loss}
        if self.table_context is None:
            self.table_context = tp.TableContext(headers=['epoch', 'fold', 'Train Loss', 'Valid Loss', 'Time'])
            self.table_context.__enter__()
        
        # if (self.current_epoch + 1) % 10 == 0:
        self.table_context([self.current_epoch+1, self.fold, self.avg_train_loss, self.avg_valid_loss, time_int])
        self.logger.log_metrics(metrics)

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.table_context.__exit__()

    
    def format_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_rounded = int(round((elapsed_time)))
        return str(datetime.timedelta(seconds=elapsed_rounded))


