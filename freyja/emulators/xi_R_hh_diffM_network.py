import torch
import torch.nn as nn
import pytorch_lightning as pl


"""
Configuration and Hyperparameters for the Halo Beta Emulator.
"""

HP = {
    # Data Constraints
    "logM_cut_max": 14.0,  # Maximum log mass
    "r_cut_max": 30.0,  # Maximum scale in Mpc/h
    "r_cut_min": 0.5,  # Minimum scale
    "n_models": 59,  # Number of simulation models available
    # Neural Network Architecture
    "batch_size": 128,  # Larger batch size stabilizes gradients
    "learning_rate": 1e-3,  # Initial LR
    "hidden_layers": [256, 256],  # Deep MLP
    "activation": "GELU",  # Smoother than ReLU, better for physics
    "dropout": 0.0,  # Regularization
    "weight_decay": 1e-4,  # L2 Regularization
    "max_epochs": 9999,
    "patience": 50,  # Early stopping patience
    # Loss Configuration
    "use_weighted_loss": True,  # If True, uses 1/sigma^2 weights
    "loss_epsilon": 1e-6,  # Stability floor for variance
}


class BetaNet(pl.LightningModule):
    def __init__(self, input_dim, output_dim, scalers, hp):
        super().__init__()
        self.save_hyperparameters(hp)
        self.scalers = scalers

        # Build MLP
        layers = []
        in_dim = input_dim
        for h_dim in self.hparams.hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))

            # Activation
            if self.hparams.activation == "GELU":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())

            # Dropout - ALWAYS append to match trained checkpoint structure
            # (Even if dropout is 0.0, the layer object must exist)
            layers.append(nn.Dropout(self.hparams.dropout))

            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)

        if self.hparams.use_weighted_loss:
            diff = (y_pred - y) ** 2
            loss = torch.mean(w * diff)
        else:
            loss = torch.nn.functional.mse_loss(y_pred, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_pred = self(x)

        mse = torch.nn.functional.mse_loss(y_pred, y)
        self.log("val_mse", mse, prog_bar=True)

        if self.hparams.use_weighted_loss:
            loss = torch.mean(w * (y_pred - y) ** 2)
            self.log("val_weighted_loss", loss)

        return mse

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_mse"},
        }
