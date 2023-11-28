import PIL
import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import (
    clip_grad_norm_, clip_grad_value_
)
from torchvision.transforms import (
    ToTensor,
)
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import (
    inf_loop,
    MetricTracker,
)
from src.utils import optional_autocast


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer_generator,
        optimizer_discriminator,
        config,
        device,
        dataloaders,
        log_step=400,  # how often WANDB will log
        log_predictions_step_epoch=5,
        mixed_precision=False,
        scheduler_generator=None,
        scheduler_discriminator=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(
            model,
            criterion,
            metrics,
            [
                optimizer_generator,
                optimizer_discriminator,
            ],
            config,
            device,
            [
                scheduler_generator,
                scheduler_discriminator,
            ],
        )
        self.skip_oom = skip_oom
        self.train_dataloader = dataloaders["train"]
        self.config = config
        self.samplerate = 22050
        if len_epoch is None:
            self.len_epoch = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.scheduler_generator = scheduler_generator
        self.scheduler_discriminator = scheduler_discriminator
        self.log_step = log_step
        self.log_predictions_step_epoch = log_predictions_step_epoch
        self.mixed_precision = mixed_precision
        self.train_metrics = MetricTracker(
            "discriminator_loss",
            "generator_loss",
            "grad norm",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )
        self.scaler = GradScaler(init_scale=8192, enabled=self.mixed_precision)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        for tensor_name in ["waves"]:
            if tensor_name in batch:
                batch[tensor_name] = batch[tensor_name].to(device)

        return batch

    def _clip_grad_norm(self, optimizer):
        self.scaler.unscale_(optimizer)
        if self.config["trainer"].get("grad_norm_clip") is not None:
            try:
                clip_grad_value_(
                    parameters=self.model.parameters(),
                    clip_value=1000.
                )
                clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.config["trainer"]["grad_norm_clip"],
                    error_if_nonfinite=True
                )
            except RuntimeError:
                return False
        return True

    def _train_epoch(self, epoch):
        self.model.train()
        self.criterion.train()
        self.train_metrics.reset()
        batch_idx = 0
        for i, batch in enumerate(
            tqdm(
                self.train_dataloader,
                desc="train",
                total=self.len_epoch,
            )
        ):
            try:
                batch = self.process_batch(
                    batch,
                    batch_idx=batch_idx,  #
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            for loss_type in [
                "generator_loss",
                "discriminator_loss",
            ]:
                self.train_metrics.update(
                    loss_type,
                    batch.get(loss_type, torch.tensor(torch.nan)).detach().cpu().item(),
                )
            self.train_metrics.update(
                "grad norm",
                self.get_grad_norm(),
            )
            if batch_idx == 0:
                last_train_metrics = self.debug(
                    batch,
                    batch_idx,
                    epoch,
                )
            elif batch_idx >= self.len_epoch:
                break
            batch_idx += 1
        self.scheduler_generator.step()
        self.scheduler_discriminator.step()
        log = last_train_metrics
        if epoch % self.log_predictions_step_epoch == 0:
            print("Logging predictions!")
            self._log_predictions()
        return log

    @torch.no_grad()
    def debug(self, batch, batch_idx, epoch):
        if self.writer is None:
            return
        self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.writer.add_scalar(
            "epoch",
            epoch,
        )
        self.logger.debug(
            "Train Epoch: {} {} Loss D/G: {:.4f}/{:.4f}".format(
                epoch,
                self._progress(batch_idx),
                self.train_metrics.avg("discriminator_loss"),
                self.train_metrics.avg("generator_loss"),
            )
        )
        self.writer.add_scalar(
            "learning rate generator",
            self.optimizer_generator.state_dict()["param_groups"][0]["lr"],
        )
        self.writer.add_scalar(
            "scaler factor",
            self.scaler.get_scale()
        )
        self.writer.add_scalar(
            "learning rate discriminator",
            self.optimizer_discriminator.state_dict()["param_groups"][0]["lr"],
        )
        audio_generator_example = (
            batch["wave_fake_detached"][0].cpu().to(torch.float32).numpy().flatten()
        )
        audio_true_example = (
            batch["wave_true"][0].cpu().to(torch.float32).numpy().flatten()
        )
        self.writer.add_audio(
            "generator",
            audio_generator_example,
            sample_rate=22050,
        )
        self.writer.add_audio(
            "true",
            audio_true_example,
            sample_rate=22050,
        )

        # try:
        #     self._log_spectrogram(batch)
        # except Exception as e:
        #     print(f"Error displaying spectrogram: {e}. Continue.")
        self._log_scalars(self.train_metrics)
        last_train_metrics = self.train_metrics.result()
        self.train_metrics.reset()
        return last_train_metrics

    def process_batch(
        self,
        batch,
        batch_idx: int,
        metrics: MetricTracker,
    ):
        batch = self.move_batch_to_device(batch, self.device)
        with torch.no_grad():
            batch.update(self.model(**batch))
        with optional_autocast(enabled=self.mixed_precision):
            batch.update(self.model.generator(**batch))

            # Discriminator
            self.optimizer_discriminator.zero_grad(set_to_none=True)

            batch.update(self.model.mp_discriminator(**batch, detach=True))
            batch.update(self.model.ms_discriminator(**batch, detach=True))
            batch.update(self.criterion.discriminator_loss(**batch))

        self.scaler.scale(batch["discriminator_loss"]).backward()
        if not self._clip_grad_norm(self.optimizer_discriminator):
            print("NaN gradients. Skipping batch")
            self.scaler.update()
            return batch
        self.scaler.step(self.optimizer_discriminator)
        self.scaler.update()
        with optional_autocast(enabled=self.mixed_precision):
            # Generator
            self.optimizer_generator.zero_grad(set_to_none=True)
            batch.update(
                self.model.mp_discriminator(
                    **batch,
                    detach=False,
                )
            )
            batch.update(
                self.model.ms_discriminator(
                    **batch,
                    detach=False,
                )
            )

            batch.update(self.criterion.generator_loss(**batch))
        self.scaler.scale(batch["generator_loss"]).backward()
        if not self._clip_grad_norm(self.optimizer_generator):
            print("NaN gradients. Skipping batch")
            self.scaler.update()
            return batch
        self.scaler.step(self.optimizer_generator)
        self.scaler.update()

        metrics.update(
            "discriminator_loss",
            batch["discriminator_loss"].item(),
        )
        metrics.update(
            "generator_loss",
            batch["generator_loss"].item(),
        )
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(
            self.train_dataloader,
            "n_samples",
        ):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(
            current,
            total,
            100.0 * current / total,
        )

    @torch.no_grad()
    def _log_predictions(self):
        self.model.eval()
        rows = {}

    @staticmethod
    def make_image(buff):
        return ToTensor()(PIL.Image.open(buff))

    @torch.no_grad()
    def _log_spectrogram(self, batch):
        spectrogram_types = [
            "_true",
            "_fake",
        ]
        for spectrogram_type in spectrogram_types:
            spectrogram = (
                batch[f"mel{spectrogram_type}"][0]
                .detach()
                .cpu()
                .to(torch.float64)
                .transpose(1, 0)
            )
            spectrogram = torch.nan_to_num(spectrogram)
            buf = plot_spectrogram_to_buf(spectrogram)
            self.writer.add_image(
                f"spectrogram_{spectrogram_type}",
                Trainer.make_image(buf),
            )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(
                        # nan occurs in first batch in first run with grad scaler
                        torch.nan_to_num(
                            p.grad,
                            nan=0,
                        ).detach(),
                        norm_type,
                    ).cpu()
                    for p in parameters
                ]
            ),
            norm_type,
        )
        return total_norm.item()

    @torch.no_grad()
    def _log_scalars(
        self,
        metric_tracker: MetricTracker,
    ):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                metric_name,
                metric_tracker.avg(metric_name),
            )
