# Deep Covariance Alignment (DCA)
Code for TGRS 2022 paper

## Getting Started

#### Requirements:
- pytorch >= 1.7.0
- python >=3.6
- pandas >= 1.1.5
### Prepare LoveDA Dataset

```bash
ln -s </path/to/LoveDA> ./LoveDA

### Evaluate DCA Model on the predict set
#### 1. Download the pre-trained [<b>weights</b>](https://drive.google.com/drive/folders/1xFn1d8a4Hv4il52hLCzjEy_TY31RdRtg?usp=sharing)
#### 2. Move weight file to log directory
```bash
mkdir -vp ./log/
mv ./URBAN100000.pth ./log/URBAN10000.pth
```

#### 3. Evaluate on test set
```bash 
bash ./scripts/predict_cbst.sh
```
Submit your test results on [LoveDA Unsupervised Domain Adaptation Challenge](https://codalab.lisn.upsaclay.fr/competitions/424) and you will get your Test score.
