import os
import argparse
import collections
import warnings

import numpy as np
import torch

import src.loss as module_loss
import src.model as module_arch
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
import random

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")
    # torch.autograd.set_detect_anomaly(True)
    # setup data_loader instances
    dataloaders = get_dataloaders(config, None)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    print(device, device_ids)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    if config["trainer"].get("compile", False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            # loss_module = torch.compile(loss_module, mode="reduce-overhead")
        except Exception:
            print("Could not compile loss")
    metrics = config['metrics']

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    optimizer_generator = config.init_obj(
        config["optimizer"],
        torch.optim,
        filter(lambda p: p.requires_grad, model.generator.parameters()),
    )
    optimizer_discriminator = config.init_obj(
        config["optimizer"],
        torch.optim,
        list(filter(lambda p: p.requires_grad, model.mp_discriminator.parameters()))
        + list(filter(lambda p: p.requires_grad, model.ms_discriminator.parameters())),
    )
    scheduler_generator = config.init_obj(
        config["lr_scheduler"], torch.optim.lr_scheduler, optimizer_generator
    )
    scheduler_discriminator = config.init_obj(
        config["lr_scheduler"], torch.optim.lr_scheduler, optimizer_discriminator
    )
    print(
        "Num params discriminator MP:",
        sum(p.numel() for p in model.mp_discriminator.parameters() if p.requires_grad),
    )
    print(
        "Num params discriminator MS:",
        sum(p.numel() for p in model.ms_discriminator.parameters() if p.requires_grad),
    )
    print(
        "Num params generator:",
        sum(p.numel() for p in model.generator.parameters() if p.requires_grad),
    )
    print("Num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer_generator,
        optimizer_discriminator,
        config=config,
        device=device,
        dataloaders=dataloaders,
        scheduler_generator=scheduler_generator,
        scheduler_discriminator=scheduler_discriminator,
        len_epoch=config["trainer"].get("len_epoch", None),
        log_step=config["trainer"].get("log_step", 200),
        log_predictions_step_epoch=config["trainer"].get(
            "log_predictions_step_epoch", 1
        ),
        mixed_precision=config["trainer"].get("mixed_precision", True),
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
