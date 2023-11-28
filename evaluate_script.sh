pip install -r requirements.txt
pip install gdown>4.7
mkdir -p default_test_model
cd default_test_model
gdown 1Cv50C8s3Qq54_lndi6AobRQlljUCExLl -O checkpoint.pth
gdown 1YNkqjKbgz3GzqN5NNUQ5q9suPvMXydOf -O config.json
cd ..
