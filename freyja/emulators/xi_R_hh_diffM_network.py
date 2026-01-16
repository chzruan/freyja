"""
The Neural Network architecture of halo bias Beta emulators using Lightning.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..config import HP


class BetaNet(pl.LightningModule):
    def __init__(self, input_dim, output_dim, scalers):
        super().__init__()
        self.save_hyperparameters()
        self.scalers = scalers

        # Build MLP
        layers = []
        in_dim = input_dim
        for h_dim in HP["hidden_layers"]:
            layers.append(nn.Linear(in_dim, h_dim))
            if HP["activation"] == "GELU":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(HP["dropout"]))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)

        if HP["use_weighted_loss"]:
            diff = (y_pred - y) ** 2
            loss = torch.mean(w * diff)
        else:
            loss = torch.nn.functional.mse_loss(y_pred, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)

        # Standard MSE for metrics
        mse = torch.nn.functional.mse_loss(y_pred, y)
        self.log("val_mse", mse, prog_bar=True)

        if HP["use_weighted_loss"]:
            loss = torch.mean(w * (y_pred - y) ** 2)
            self.log("val_weighted_loss", loss)

        return mse

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=HP["learning_rate"], weight_decay=HP["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_mse"},
        }
