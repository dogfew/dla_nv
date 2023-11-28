import argparse
import json
import os
from pathlib import Path

import torch
import src.model as module_model
from src.utils import ROOT_PATH
from src.utils.parse_config import ConfigParser
from src.waveglow import utils
from src.synthesis.synthesis import Synthesizer
import warnings
import os
import shutil

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
    waveglow = utils.get_WaveGlow().cuda()
    output_dir = args.out_dir
    synthesizer = Synthesizer(waveglow=waveglow, device=device, dir=output_dir)
    with torch.no_grad():
        if args.text is not None:
            raw_text = args.text
            if not args.arpa_input:
                preprocessed_phonemes = synthesizer.g2p(raw_text)
                preprocessed_phonemes = [item.replace('.', ' ')
                                         for item in preprocessed_phonemes if item != ' ']
            else:
                preprocessed_phonemes = raw_text.split(' ')
                preprocessed_phonemes = [i if i != ' ' else '' for i in preprocessed_phonemes]
            synthesizer(model, preprocessed_phonemes, idx='custom', alpha=args.speed, beta=args.pitch, gamma=args.energy)
        else:
            synthesizer.create_audios(model)
    os.makedirs(os.path.join(output_dir, 'waveglow'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'default'), exist_ok=True)
    files = os.listdir(output_dir)
    for filename in files:
        source_path = os.path.join(output_dir, filename)
        if os.path.isdir(source_path):
            continue
        if filename.endswith("waveglow.wav"):
            destination_path = os.path.join(output_dir, 'waveglow', filename)
        else:
            destination_path = os.path.join(output_dir, 'default', filename)
        if os.path.exists(destination_path):
            os.remove(destination_path)

        shutil.move(source_path, destination_path)


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
        "-o",
        "--out_dir",
        default="final_results",
        type=str,
        help="Output directory for results (default: final_results)",
    )
    args.add_argument(
        "-p",
        "--pitch",
        default=1.0,
        type=float,
        help="Pitch adjustment factor (default: 1.0)",
    )
    args.add_argument(
        "-s",
        "--speed",
        default=1.0,
        type=float,
        help="Speed adjustment factor (default: 1.0)",
    )
    args.add_argument(
        "--arpa_input",
        default=False,
        type=bool,
        help="Whether to use ARPA in custom text."
    )
    args.add_argument(
        "-e",
        "--energy",
        default=1.0,
        type=float,
        help="Energy adjustment factor (default: 1.0)",
    )
    args.add_argument(
        '--text',
        default=None,
        type=str,
        help="Text to speech."
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
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
