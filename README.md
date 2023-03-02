# Deep Covariance Alignment (DCA)
Code for TGRS 2022 paper, [**"Deep Covariance Alignment for Domain Adaptive Remote Sensing Image Segmentation"**](https://ieeexplore.ieee.org/document/9745130), accepted.

Authors: Linshan Wu, Ming Lu, <a href="https://www.leyuanfang.com/">Leyuan Fang</a>

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


## Evaluate DCA Model on the test set
### 1. Download the pre-trained [<b>weights</b>](https://drive.google.com/drive/folders/1oenWpYADqd-tTx7JeDQknxRNd3mgW2kQ)
### 2. Move weight file to log directory
```bash
mkdir -vp ./log/
mv ./URBAN_0.4635.pth ./log/URBAN_0.4635.pth
mv ./RURAL_0.4517.pth ./log/RURAL_0.4517.pth
python My_test.py
```

### 3. Evaluate on the website
Submit your test results on [LoveDA Unsupervised Domain Adaptation Challenge](https://codalab.lisn.upsaclay.fr/competitions/424) and you will get your Test score.

Or you can download our [<b>results</b>](https://drive.google.com/drive/folders/1WQdgveVwW016BMKvw1Afj6o_MQ9UcZeA)
## Train DCA Model
```bash 
python DCA_train.py
```
The training [<b>logs</b>](https://drive.google.com/drive/folders/1oenWpYADqd-tTx7JeDQknxRNd3mgW2kQ)

## Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```
@ARTICLE{9745130,
  author={Wu, Linshan and Lu, Ming and Fang, Leyuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Deep Covariance Alignment for Domain Adaptive Remote Sensing Image Segmentation}, 
  year={2022},
  volume={60},
  number={},
  pages={1-11},
  doi={10.1109/TGRS.2022.3163278}}
```

For any questions, please contact [Linshan Wu](mailto:15274891948@163.com).
