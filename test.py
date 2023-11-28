import argparse
import json
from pathlib import Path

import numpy as np
import torchaudio
import torch
from tqdm import tqdm

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
import warnings
import os

warnings.filterwarnings("ignore")

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main(config, args):
    logger = config.get_logger("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    output_dir = args.out_dir
    with torch.no_grad():
        if args.mel_dir is not None:
            print(f"Processing mel files from {args.mel_dir}")
            for mel_file in filter(
                lambda f: f.endswith(".npy"), os.listdir(args.mel_dir)
            ):
                mel_path = os.path.join(args.mel_dir, mel_file)
                mel_data = np.load(mel_path)
                mel_tensor = torch.tensor(mel_data).to(device)
                generated_audio = (
                    model.generator(mel_tensor)["wave_fake"].cpu().view(1, -1)
                )
                output_filename = os.path.splitext(mel_file)[0] + ".wav"
                torchaudio.save(
                    os.path.join(args.out_dir, output_filename),
                    generated_audio,
                    sample_rate=args.sample_rate,
                    format="wav",
                )
        elif args.audio_dir is not None:
            print(f"Processing audios from {args.audio_dir}")
            os.makedirs(os.path.join(output_dir), exist_ok=True)
            for i, audio_file in enumerate(
                list(filter(lambda f: f.endswith(".wav"), os.listdir(args.audio_dir)))
            ):
                batch = {
                    "waves": torchaudio.load(os.path.join(args.audio_dir, audio_file))[
                        0
                    ]
                }
                batch = Trainer.move_batch_to_device(batch, device)
                batch.update(model(**batch))
                batch.update(model.generator(**batch))
                generated_audio = batch["wave_fake"].cpu().view(1, -1)
                output_filename = os.path.splitext(audio_file)[0] + ".wav"
                torchaudio.save(
                    os.path.join(args.out_dir, output_filename),
                    generated_audio,
                    sample_rate=args.sample_rate,
                    format="wav",
                )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Test")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "--sample_rate",
        default=22050,
        type=int,
        help="sample rate of audio to generate",
    )
    args.add_argument(
        "-o",
        "--out_dir",
        default="final_results",
        type=str,
        help="Output directory for results (default: final_results)",
    )
    args.add_argument("--text", default=None, type=str, help="Text to speech.")
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-m",
        "--mel_dir",
        default=None,
        type=str,
        help="Directory with mel spectrograms",
    )
    args.add_argument(
        "-a",
        "--audio_dir",
        default="test_data",
        type=str,
        help="Directory with audios",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    args = args.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))
    main(config, args)
