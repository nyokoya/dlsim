# Deep Learning from Simulated Data for Flood and Debris Flow Mapping

Code for the paper [Breaking the Limits of Remote Sensing by Simulation and Deep Learning for Flood and Debris Flow Mapping](https://arxiv.org/abs/2006.05180).
![alt text](./data/dlsim_overview.png?raw=true)

## Installation
If necessary create a new Python environment.
Install requirements.
```
pip install -r requirements.txt
```


## Dataset Preparation
The datasets presented in the paper (Northern Kyushu 2017 and Western Japan 2018) can be download [here](https://drive.google.com/file/d/1G3TWcCaEzDAUcF_QpX50qA5WDq0O1sZv/view?usp=sharing). Please unzip the file in the directory `data`.


## Training and Testing
```bash
# To train on the Northern Kyushu 2017 dataset for flood mapping, for example.
python dlsim.py --data NK2017 --train_test train --type wl

# To test pretrained models
python dlsim.py --data NK2017 --train_test test --type wl
```


## Citation
```
@article{yokoya2020dlsim,
  title={Breaking the Limits of Remote Sensing by Simulation and Deep Learning for Flood and Debris Flow Mapping},
  author={Yokoya, Naoto and Yamanoi, Kazuki and He, Wei and Baier, Gerald and Adriano, Bruno and Miura, Hiroyuki and Oishi, Satoru},
  journal={arXiv:2006.05180},
  year={2020}
}
```
