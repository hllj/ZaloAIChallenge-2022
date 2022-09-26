from typing import Any

import torch
import torch.nn as nn

# import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.optimizer import Optimizer

from .loss import AdversarialLoss
from .networks import DepthEstimationNet, Discriminator, HazeProduceNet, HazeRemovalNet


class Model(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type="lsgan")

        self.depth_estimator = DepthEstimationNet(*args, **kwargs)
        self.haze_producer = HazeProduceNet(*args, **kwargs)
        self.haze_remover = HazeRemovalNet(*args, **kwargs)
        self.discriminator = Discriminator(
            in_channels=3, use_spectral_norm=True, use_sigmoid=True
        )

        self.save_hyperparameters()

    def step(self, batch, batch_idx):
        pass
        # cleaned, hazed = batch
        # d_tilde = self.depth_estimator(cleaned)
        # beta_tilde = self.depth_estimator()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx) -> None:
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx) -> None:
        return super().test_step(batch, batch_idx)

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ):
        optimizer.zero_grad(set_to_none=True)
