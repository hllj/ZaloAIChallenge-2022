import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection, PeakSignalNoiseRatio

from .loss import AdversarialLoss
from .networks import DepthEstimationNet, Discriminator, HazeProduceNet, HazeRemovalNet


class Model(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.save_hyperparameters(config)

        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type="lsgan")

        self.depth_estimator = DepthEstimationNet(**self.hparams.depth_estimator)
        self.haze_producer = HazeProduceNet(**self.hparams.haze_producer)
        self.haze_remover = HazeRemovalNet(**self.hparams.haze_remover)
        self.discriminator = Discriminator(
            in_channels=3, use_spectral_norm=True, use_sigmoid=True
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
        opt_producer, opt_disc, opt_remover = self.optimizers()
        cleaned, hazed = batch  # Unpaired

        d_tilde = self.depth_estimator(cleaned)
        hazed_synth = self.haze_producer(cleaned, d_tilde)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_real, _ = self.discriminator(hazed)
        errD_real = self.adversarial_loss(d_real, is_disc=True, is_real=True)

        d_fake, _ = self.discriminator(hazed_synth.detach())
        errD_fake = self.adversarial_loss(d_fake, is_disc=True, is_real=False)

        errD = errD_real + errD_fake

        opt_disc.zero_grad()
        self.manual_backward(errD)
        opt_disc.step()

        #####################
        # Optimize Producer #
        #####################
        d_fake, _ = self.discriminator(hazed_synth)
        errP_gen = self.adversarial_loss(d_fake, is_disc=False, is_real=False)

        cleaned_hat = self.haze_remover(hazed_synth)
        errP_recon = -self.l1_loss(cleaned_hat, cleaned)  # We want to maximize this one

        errP = errP_gen + errP_recon

        opt_producer.zero_grad()
        self.manual_backward(errP)
        opt_producer.step()

        ####################
        # Optimize Remover #
        ####################
        cleaned_hat = self.haze_remover(hazed_synth.detach())
        errR = self.l1_loss(cleaned_hat, cleaned)

        opt_remover.zero_grad()
        self.manual_backward(errR)
        opt_remover.step()

        # Metrics calculation
        metric_dict = self.train_metrics(cleaned_hat, cleaned)

        self.log_dict(
            {
                "train/producer_loss": errP,
                "train/remover_loss": errR,
                "train/discriminator_loss": errD,
            },
            prog_bar=True,
            sync_dist=self.hparams.sync_dist,
        )
        metric_dict.update(
            {
                "train/producer_reconstruction_loss": errP_recon,
                "train/producer_generator_loss": errP_gen,
            }
        )
        self.log_dict(metric_dict, sync_dist=self.hparams.sync_dist)

    def validation_step(self, batch, batch_idx) -> None:
        cleaned, hazed = batch
        cleaned_hat = self.haze_remover(hazed)
        metric_dict = self.val_metrics(cleaned_hat, cleaned)
        self.log_dict(metric_dict, prog_bar=True, sync_dist=self.hparams.sync_dist)

        return cleaned_hat

    def test_step(self, batch, batch_idx) -> None:
        raise NotImplementedError

    def configure_optimizers(self):
        opt_producer = torch.optim.AdamW(
            [
                {"params": self.depth_estimator.parameters()},
                {"params": self.haze_producer.parameters()},
            ],
            lr=self.hparams.optimizer.opt_producer.lr,
            weight_decay=self.hparams.optimizer.opt_producer.wd,
        )
        opt_disc = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.hparams.optimizer.opt_disc.lr,
            weight_decay=self.hparams.optimizer.opt_disc.wd,
        )
        opt_remover = torch.optim.AdamW(
            self.haze_remover.parameters(),
            lr=self.hparams.optimizer.opt_remover.lr,
            weight_decay=self.hparams.optimizer.opt_remover.wd,
        )
        return opt_producer, opt_disc, opt_remover

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ):
        optimizer.zero_grad(set_to_none=True)
