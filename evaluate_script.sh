pip install -r requirements.txt
pip install gdown>4.7
mkdir -p default_test_model
cd default_test_model
gdown 1fm1WN9_7TVpzMwYUJwtir6DpPCyQNATm -O checkpoint.pth
cd ..
