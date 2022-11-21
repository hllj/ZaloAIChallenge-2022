import logging
from pathlib import Path
import random
import numpy as np
import hydra
import pytorch_lightning as pl
import torch
import wandb
import random
import numpy as np

### seed
seed = 6789
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark= False

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

from src.dataset import LivenessDatamodule
from src.model import TIMMModel

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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            n = self.samples
            cleaned, _ = batch
            cleaned_hat, hazed_synth = outputs["cleaned_hat"], outputs["hazed_synth"]
            images = [
                torch.cat([hazed_inp, cleaned_pred, cleaned_gt], dim=-1)
                for hazed_inp, cleaned_pred, cleaned_gt in zip(
                    hazed_synth[:n], cleaned_hat[:n], cleaned[:n]
                )
            ]
            captions = ["Inp - Pred - GT"] * n

            # Option 1: log images with `WandbLogger.log_image`
            self.wandb_logger.log_image(
                key="train/visualization", images=images, caption=captions
            )

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

def train(config):
    # config.seed = pl.seed_everything(seed=config.seed, workers=True)
    
    wandb_logger = WandbLogger(
        project="zalo_2022",
        log_model=False,
        settings=wandb.Settings(start_method="fork"),
        name=Path.cwd().stem,
        dir=Path.cwd()
    )

    # Create callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(**config.model_ckpt))
    callbacks.append(RichProgressBar(config.refresh_rate))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    # callbacks.append(LogPredictionSamplesCallback(wandb_logger))

    OmegaConf.set_struct(config, False)
    strategy = config.trainer.pop("strategy", None)
    OmegaConf.set_struct(config, True)
    if strategy == "ddp" and config.trainer.accelerator == "gpu":
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have

        # TODO: Currently only handles gpus = -1 or an int number
        if config.trainer.devices == -1:
            config.trainer.devices = torch.cuda.device_count()

        num_nodes = getattr(config.trainer, "num_nodes", 1)
        total_gpus = max(1, config.trainer.devices * num_nodes)
        config.dataset.batch_size = int(config.dataset.batch_size / total_gpus)
        config.dataset.num_workers = int(config.dataset.num_workers / total_gpus)
        strategy = DDPStrategy(
            find_unused_parameters=config.ddp_plugin.find_unused_params,
            gradient_as_bucket_view=True,
            ddp_comm_hook=default.fp16_compress_hook
            if config.ddp_plugin.fp16_hook
            else None,
            static_graph=config.ddp_plugin.static_graph,
        )
    model = TIMMModel(config.model)
    datamodule = LivenessDatamodule(config.dataset)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        strategy=strategy,
        **config.trainer,
    )

    wandb_logger.watch(model, log="parameters", log_graph=False)
    trainer.fit(model, datamodule=datamodule)
    wandb.finish()


@hydra.main(config_path="configs", config_name="baseline")
def main(config: DictConfig) -> None:
    log.info("Zalo AI Challenge - Liveness Detection")
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
