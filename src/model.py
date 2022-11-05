import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection, PeakSignalNoiseRatio

from .loss import LabelSmoothingLoss
from .networks import EfficientNetB3DSPlus


class TIMMModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.model = EfficientNetB3DSPlus(
            model_name=config.model_name, n_class=config.n_class
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.lr_scheduler.step_size,
            gamma=self.config.lr_scheduler.gamma,
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        x = self.model(x)
        return x

    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx) -> None:
        loss, preds, targets = self.step(batch)
        acc = self.train_acc(preds, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> None:
        val_loss, val_preds, val_targets = self.step(batch)
        val_acc = self.val_acc(val_preds, val_targets)

        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx) -> None:
        raise NotImplementedError

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()
