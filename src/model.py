import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection, PeakSignalNoiseRatio

from .loss import AdversarialLoss
from .networks import DepthEstimationNet, Discriminator, HazeProduceNet, HazeRemovalNet

# from torchvision.models import VGG11_Weights, vgg11


class Model(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.save_hyperparameters(config)

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.adversarial_loss = AdversarialLoss(loss_type="wgan-gp")

        self.depth_estimator = DepthEstimationNet(**self.hparams.depth_estimator)
        self.haze_producer = HazeProduceNet(**self.hparams.haze_producer)
        # self.loss_estimator = vgg11(weights=VGG11_Weights.DEFAULT)
        # self.loss_estimator.classifier[-1] = nn.Linear(4096, 1)
        # self.loss_estimator.classifier.append(nn.ReLU(True))
        self.haze_remover = HazeRemovalNet(**self.hparams.haze_remover)
        self.discriminator = Discriminator(
            in_channels=3, use_spectral_norm=True, use_sigmoid=False
        )

        metric_collection = MetricCollection(
            [
                PeakSignalNoiseRatio(data_range=1.0),
                # Using SSIM will cause OOM
            ],
        )
        self.train_metrics = metric_collection.clone("train/")
        self.val_metrics = metric_collection.clone("val/")

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx) -> None:
        pass

    def validation_step(self, batch, batch_idx) -> None:
        pass

    def test_step(self, batch, batch_idx) -> None:
        raise NotImplementedError

    def configure_optimizers(self):
        pass

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ):
        optimizer.zero_grad(set_to_none=True)
