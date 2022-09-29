import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

from src.dataset import DehazeDatamodule
from src.model import Model

log = logging.getLogger(__name__)


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


class LogPredictionSamplesCallback(Callback):
    def __init__(self, wandb_logger, samples=8) -> None:
        super().__init__()

        self.wandb_logger: WandbLogger = wandb_logger
        self.samples = samples

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log image predictions from the first batch
        if batch_idx == 0:
            n = self.samples
            cleaned, hazed = batch
            cleaned_hat = outputs
            images = [
                torch.cat([hazed_inp, cleaned_pred, cleaned_gt], dim=-1)
                for hazed_inp, cleaned_pred, cleaned_gt in zip(
                    hazed[:n], cleaned_hat[:n], cleaned[:n]
                )
            ]
            captions = ["Inp - Pred - GT"] * n

            # Option 1: log images with `WandbLogger.log_image`
            self.wandb_logger.log_image(
                key="val/visualization", images=images, caption=captions
            )

            # Option 2: log images and predictions as a W&B Table
            # columns = ["image", "ground truth", "prediction"]
            # data = [
            #     [wandb.Image(x_i), y_i, y_pred]
            #     for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))
            # ]
            # self.wandb_logger.log_table(
            #     key="sample_table", columns=columns, data=data
            # )


def train(config):
    config.seed = pl.seed_everything(seed=config.seed, workers=True)

    wandb_logger = WandbLogger(
        project="learn-to-dehaze",
        log_model=False,
        settings=wandb.Settings(start_method="fork"),
        name=Path.cwd().stem,
    )

    # Create callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(**config.model_ckpt))
    callbacks.append(RichProgressBar(config.refresh_rate))
    callbacks.append(LogPredictionSamplesCallback(wandb_logger))

    OmegaConf.set_struct(config, False)
    strategy = config.trainer.pop("strategy", None)
    OmegaConf.set_struct(config, True)
    if strategy == "ddp":
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have

        # TODO: Currently only handles gpus = -1 or an int number
        if config.trainer.gpus == -1:
            config.trainer.gpus = torch.cuda.device_count()

        num_nodes = getattr(config.trainer, "num_nodes", 1)
        total_gpus = max(1, config.trainer.gpus * num_nodes)
        config.dataset.batch_size = int(config.dataset.batch_size / total_gpus)
        config.dataset.num_workers = int(config.dataset.num_workers / total_gpus)
        strategy = DDPPlugin(
            find_unused_parameters=config.ddp_plugin.find_unused_params,
            gradient_as_bucket_view=True,
            ddp_comm_hook=default.fp16_compress_hook
            if config.ddp_plugin.fp16_hook
            else None,
        )

    model = Model(config.model)
    datamodule = DehazeDatamodule(config.dataset)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        strategy=strategy,
        **config.trainer,
    )

    wandb_logger.watch(model, log_graph=False)
    trainer.fit(model, datamodule=datamodule)


@hydra.main(config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    log.info("Learn to Dehaze")
    log.info(f"Current working directory : {Path.cwd()}")

    if config.state == "train":
        set_debug_apis(state=False)
        train(config)
    elif config.state == "debug":
        pass
    elif config.state == "test":
        set_debug_apis(state=False)
        pass


if __name__ == "__main__":
    main()
