## BPMI
Code for the SGNS and LogitSGNS models from the paper [Binarized PMI Matrix: Bridging Word Embeddings and Hyperbolic Spaces](https://www.overleaf.com/read/tjnppdygcyxz)

### Requirements
Code is written in Python 3.6 and requires Pytorch 1.3+. It also requires the following Python modules: `numpy`, `gensim`, `argparse`. You can install them via:
```
sudo pip3.6 install numpy gensim argparse
```

### Data
Data should be put into the `data/` directory. You may use `text8` as an example:
```
mkdir data
cd data
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
cd ..
```

### Model
To reproduce the SGNS results from Table 1
```
mkdir embeddings
python3.6 main.py
python3.6 eval.py
```
