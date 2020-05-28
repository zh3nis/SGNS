## SGNS
Code for the SGNS and LogitSGNS models from the paper [Binarized PMI Matrix: Bridging Word Embeddings and Hyperbolic Spaces](https://arxiv.org/abs/2002.12005)

### Requirements
Code is written in Python 3.6 and requires Pytorch 1.3+. It also requires the following Python modules: `numpy`, `gensim`, `argparse`, `gdown`. You can install them via:
```bash
pip install numpy gensim argparse gdown
```

### Data
Data should be put into the `data/` directory. You may use `text8` as an example:
```bash
mkdir data
cd data
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip

gdown https://drive.google.com/uc?id=1oZk6vhkn4-0hznirVqVeIGtm9b53XO5g
unzip valid.zip

cd ..
```
(`text8` are the first 100MB of Wikipedia, `valid.txt` are the next 10MB of Wikipedia)

### Model
To reproduce the SGNS results from Table 1
```
mkdir embeddings
python main.py --valid data/valid.txt
python eval.py
```
