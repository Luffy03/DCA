# Deep Covariance Alignment (DCA)
Code for TGRS 2022 paper, **"Deep Covariance Alignment for Domain Adaptive Remote Sensing Image Segmentation"**, accepted.
Linshan Wu, Ming Lu, <a href=https://sites.google.com/site/leyuanfang/home">Leyuan Fang</a>

This repository highly depends on the <a href="https://github.com/Junjue-Wang/LoveDA">LoveDA</a> repository implemented by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>. We thank the authors for their great work and clean code. Appreciate it!
## Getting Started

#### Requirements:
- pytorch >= 1.7.0
- python >=3.6
- pandas >= 1.1.5

### Install Ever + Segmentation Models PyTorch + audtorch
```bash
pip install ever-beta==0.2.3
pip install git+https://github.com/qubvel/segmentation_models.pytorch
pip install audtorch
```


### Prepare LoveDA Dataset

```bash
ln -s </path/to/LoveDA> ./LoveDA
```


## Evaluate DCA Model on the predict set
### 1. Download the pre-trained [<b>weights</b>](https://drive.google.com/drive/folders/1oenWpYADqd-tTx7JeDQknxRNd3mgW2kQ)
### 2. Move weight file to log directory
```bash
mkdir -vp ./log/
mv ./URBAN_0.4635.pth ./log/URBAN_0.4635.pth
mv ./RURAL_0.4517.pth ./log/RURAL_0.4517.pth
```

### 3. Evaluate on Urban test set
Submit your test results on [LoveDA Unsupervised Domain Adaptation Challenge](https://codalab.lisn.upsaclay.fr/competitions/424) and you will get your Test score.

## Train DCA Model
```bash 
python DCA_train.py
```
The training [<b>logs</b>](https://drive.google.com/drive/folders/1oenWpYADqd-tTx7JeDQknxRNd3mgW2kQ)

## Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```
waiting
```

For any questions, please contact Linshan Wu.
