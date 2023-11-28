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
gdown 1Cv50C8s3Qq54_lndi6AobRQlljUCExLl -O checkpoint.pth
gdown 1YNkqjKbgz3GzqN5NNUQ5q9suPvMXydOf -O config.json
cd ..
```

## Speech synthesis

### Generate sentences required in task
You can just run this script and check audios in `final_results/waveglow`
```shell
python test.py +resume=<checkpoint_path> test_settings.mel_dir=None 
```

## Training
To prepare data, run: 
```shell
pip install -r requirements.txt
bash prep_script.sh
```

To reproduce my final model, train FastSpeech2 with this config (400 epochs): 
```shell
python train.py data.train.batch_size=16 data.train.datasets.0.args.max_len=45056
```

**Optional Tasks:**

- (up to +1.5) MFA alignments (in textgrid format) which are downloadable with `prep_script.sh`.
Alignments are created in a script `generate_data_mfc.py` and the final model
is trained on them. 
