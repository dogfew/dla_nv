# Text to Speech project

(Also, you can check `colab_notebook.ipynb` file, which contains commands for installation, speech synthesis, and training, and is ready to run in Google Colab)

## Installation

Make sure that your python version >= 3.10

Run commands in `evaluate_script.sh`
```shell 
bash evaluate_script.sh
```
The commands in file `evaluate_script.sh` are: 
```shell
pip install -r requirements.txt
pip install gdown>4.7
mkdir -p default_test_model
cd default_test_model
gdown 1fm1WN9_7TVpzMwYUJwtir6DpPCyQNATm -O checkpoint.pth
cd ..
```

## Speech synthesis

### Generate sentences required in task
You can just run this script and check audios in `final_results/waveglow`
If you have ready audios, run: 
```shell
python test.py +resume="default_test_model/checkpoint.pth" test_settings.out_dir="final_results" test_settings.audio_dir="test_data"
```

If you have ready mels in `.npy` format: 
```shell 
python test.py +resume="default_test_model/checkpoint.pth" test_settings.out_dir="final_results" test_settings.mel_dir="mel_test_data"
```

## Training
To prepare data, run: 
```shell
pip install -r requirements.txt
bash prep_script.sh
```

To reproduce my final model, train Hi-Fi GAN. Config path: `src/configs/config.yaml`: 
```shell
python train.py
```

**Optional Tasks:**

- (up to +1) for Hydra. There were major changes in `src/utils/parse_config.py`, `train.py` and `test.py` files
