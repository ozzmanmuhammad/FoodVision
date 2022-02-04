# FoodVision
Deep learning project for the image classification of 10 - 101 food classes.

## Overview
For this project I built a food classifier to identify food from different food images. This project aim was to beat 77.4% top-1 accuracy of [DeepFood paper](https://arxiv.org/abs/1606.05675).
I was able to get the model to predict the class of the food from 101 classes with 80% accuracy after minimal tuning. To get these results I used transfer learning (no fine-tuning) on a existing noncomplex pretrained CNN model "EfficienetNetB0". This created time efficiencies and solid results. Further experiments are needed to improve the accuracy like using more complex pretrained models (EfficienetNetB4 or ResNet34) or fine-tuning the top-layers.

## Code and Resources Used
Python Version: 3.8
Tensorflow Version: 2.7.0
Packages: pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle
GPU Configuration: Using Google Colab
