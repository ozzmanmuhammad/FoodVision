# FoodVision
Deep learning project for the image classification starting from 10 classes to 101 food classes.

## Overview
For this project I built a food classifier to identify food from different food images. This project aim was to beat 77.4% top-1 accuracy of [DeepFood paper](https://arxiv.org/abs/1606.05675).
I was able to get the model to predict the class of the food from 101 classes with 70% accuracy with out fine-tuning and 80% with fine-tuned model. To get these results I used transfer learning (no fine-tuning) on a existing noncomplex pretrained (on ImageNet) CNN model "EfficienetNetB0". This created time efficiencies and solid results. Further experiments are needed to improve the accuracy like using more complex pretrained models (EfficienetNetB4 or ResNet34).

## Code and Resources Used
- Python Version: 3.8
- Tensorflow Version: 2.7.0
- Packages: pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle
- GPU Configuration: Using Google Colab

## Dataset and EDA
The Food101 datasets was downloaded from [Kaggle](https://www.kaggle.com/dansbecker/food-101).

|               | Exp #1: 10-Classes  | Exp #2: 101-Classes  |
| ------------- |:-------------------:|:--------------------:|
| Dataset source| Preprocessed download from Kaggle| TensorFlow Datasets |
| Train data| 7,575 images|75,750 images |
| Test data | 25,250 images|25,250 images|
| Data loading|tf pre-built function|tf.data API|
|Target results|50.76% top-1 accuracy (beat [Food101 paper](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf))|77.4% top-1 accuracy (beat [DeepFood paper](https://arxiv.org/abs/1606.05675))|

some examples from the 101Food datasets:

![alt text](https://github.com/ozzmanmuhammad/FoodVision/blob/main/Images/Food101_examples.png "Train data examples")

## Model Building and Performance

|               | Exp #1: 10-Classes  | Exp #2: 101-Classes  |
| ------------- |:-------------------:|:--------------------:|
| Models | EfficienetNetB0| EfficienetNetB0 |
| Pretrained| on ImageNet |in ImageNet |
| Fine-Tuned | Yes | Yes |
| Fine-Tuned layers | last 5 | all |
| Accuracy |~57%|~80%|
| Training Time |~5m|~20m|





