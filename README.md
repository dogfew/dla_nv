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
python test.py
```

### Generate custom sentences 
You can pass custom text and create audio with different speed, pitch and energy params. 

**Note:** your text will be processed using `g2p` model from `g2p_en` package, so it can incorrectly spell
some non-english surnames etc.
```shell
python test.py --text "I am Vladimir Pere pelkin and this model is named Fast Speech Second" --speed 0.9 --energy 0.9 --pitch 0.9 -o "your_dir"
```

The out will be here: 
```
your_dir/waveglow/speed=0.9_pitch=0.9_energy=0.9_id=custom_waveglow.wav
```
Also you can use arpa phonemes for speech generation with argument `--arpa_input True`
```shell
python test.py --text "Y AA1 N EH1 HH OW1 CH UW1 B OW1 L SH IH0 D OW1 M AA1 SH EH0 K P OW1 G L UW1 B IH1 N N OW1 M UW1 UW1 OW B UW1 CH EH1 N IY0 UW1 V OW1 B R AA1 B OW1 T T K EH1 Z V UW1 K AH0 AH0" --arpa_input True -o "arpa_results"
```
You can find output file here: 
```angular2html
arpa_results/waveglow/speed=1.0_pitch=1.0_energy=1.0_id=custom_waveglow.wav
```

Using arpa, you even can generate zombie voice!
```shell
python test.py --arpa_input True --text "spn spn spn spn spn spn spn spn" --out_dir "zombie voice"
```

## Training
To prepare data, run: 
```shell
pip install -r requirements.txt
bash prep_script.sh
```

To reproduce my final model, train FastSpeech2 with this config (400 epochs): 
```shell
python train.py -c src/configs/config_fastspeech2.json
```

**Optional Tasks:**

- (up to +1.5) MFA alignments (in textgrid format) which are downloadable with `prep_script.sh`.
Alignments are created in a script `generate_data_mfc.py` and the final model
is trained on them. 
