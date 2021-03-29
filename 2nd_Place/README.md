# Predicting Wind Speeds of Tropical Storms using Stack of Multi-Step CNNs

Second place solution for Wind-dependent Variables: Predict Wind Speeds of Tropical Storms.  

Note: Pre-computed model weights (`archive.zip`) can be downloaded to the top level directory from `s3://drivendata-competition-radiant-earth/2nd_place/`.

## Summary of Approach

My approach has three stages. In the first stage, ImageNet [1] pretrained CNNs are finetuned on the dataset with different time steps in multiple of 3s. Each 3 time step images are concatenated channel wise and pass through the CNN for feature extraction. The features of each 3 time steps are concatenated and pass through a fully connected layer for final output. Some models are trained (mostly ResNet-50 [2] backbone) with 1, 3, 6, 9, and 12 consecutive time steps and some models with 6 time steps and spacing more than 1 (for example in case of spacing 2 images 1,3,5,7,9,11 are used instead of images 1,2,3,4,5,6). Aside from rotation and flipping, time augmentation is applied by dropping or repeating one of the input images (except for the main image). All models are trained using 224X224 image size, 5 group-folds, Adam optimizer [3], cosine annealing schedculer [4] and test time augmentation is applied on their predictions.

In the second stage, around 200 models are trained on the output predictions of first stage models and taking into consideration a history of 25-30 time steps. The models are combination of:
* Xgboost [5]
* 1D CNN
* LSTM [6]
* GRU [7]
* MLP
* Transformer [8]
* Decision Tree
* Linear Regression
Each model in the previous list is trained on a different first stage CNN output while using Deep Adaptive Input Normalization [9] (in neural network-based models only).

In the final stage, ensemble selection [10] is applied to combine the best group of second stage models.

## Prerequisites

Firstly, you need to have

* Ubuntu 18.04
* Python 3.6.9
* Jupyter
* 32G RAM + 64G Swap
* 11 GB GPU RAM (at least)

Secondly, you need to install the dependencies by running:

```
pip3 install -r requirements.txt
```

## Assumptions

Data files (training_set_features.csv, test_set_features.csv and submission_format.csv) and data folders (re-train-images and re-test-images) should be in the same directory with all training scripts.

## Project files

### First stage training scripts

* *conv.py: these scripts train K-fold multi-step ResNet models, save their weights, run inference on validation and test sets, and save all predictions (to be used in next stage of training). In addition, conv.py (only) saves intermediate features of all training and test images (to bes used in the next stage of training).
* *conv_inference.py: these scripts run the inference on test set only using archived multi-step ResNet models and save test predictions (to be used in next stage of training).
* prepare_dataset_v*.py: these scripts concatenate a history of time steps for each sample in the test set to be used in inference.

### Second stage training scripts

* xgb_Bulldozer.py: trains and saves validation and test predictions of 100 stacking xgboost models based on statistical features with different first stage CNN outputs and time steps.
* dt.py: trains and saves validation and test predictions of decision tree model based on statistical features with different first stage CNN outputs.
* ts_cnn.py: trains and saves validation and test predictions of 1D-CNN model based on single first stage CNN outputs, features extracted from first stage CNN trained on 1 time step, and relative time features.
* mlp.py: trains and saves validation and test predictions of multi-layer perceptron model based on single first stage CNN outputs, features extracted from first stage CNN trained on 1 time step, and relative time features.
* lstm.py: trains and saves validation and test predictions of LSTM model based on single first stage CNN outputs, features extracted from first stage CNN trained on 1 time step, and relative time features.
* gru.py: trains and saves validation and test predictions of GRU model based on single first stage CNN outputs, features extracted from first stage CNN trained on 1 time step, and relative time features.
* ts_transformer.py: trains and saves validation and test predictions of Transformer model based on single first stage CNN outputs, features extracted from first stage CNN trained on 1 time step, and relative time features.
* linear_reg.py: trains and saves validation and test predictions of linear regression model based on single first stage CNN outputs, features extracted from first stage CNN trained on 1 time step, and relative time features.
* dain.py: Deep Adaptive Input Normalization layer.

### Third stage training scripts

* es_ensemble.py: apply Ensemble Selection algorithm on all second stage models and outputs submission file (``` final_sub.csv ```).

### Submission generation scripts

* generate_submission.ipynb: a notebook with command lines to train all models and output final submission file. Each training stage starts with a headline.
* generate_submission_with_first_stage_inference_only.ipynb: a notebook with command lines to predict using archived first stage models, train second stage models and output final submission file. Each stage starts with a headline.

### Archive

* archive.zip: pre-computed weights for all first stage models can be downloaded to the top level directory from s3://drivendata-competition-radiant-earth/2nd_place/

## Running

### Training all models from scratch

run all cells in ```generate_submission.ipynb```. It should take ~5 days of training and inference (assuming the same HW specs).

### Training only second stage models

1. unzip archive.zip.
2. copy all folders from archive to the same directory as all training scripts.
3. run all cells in ```generate_submission_with_first_stage_inference_only.ipynb```.

It should take ~30 hours of training and inference (assuming the same HW specs).

### Training all models from scratch except for some first stage models

1. unzip archive.zip.
2. copy only the folders belongs to the models you don't want to re-train from archive to the same directory as all training scripts. You can find output folder name for each model commented in ```generate_submission.ipynb```.
3. run the cells of only first stage models you want to train and all the cells from second and third stage models.

Note that training scripts of first stage models will halt if they find their output folders exist in the same directory.

## References

[1] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.
[2] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
[3] Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
[4] Huang, Gao, et al. "Snapshot ensembles: Train 1, get m for free." arXiv preprint arXiv:1704.00109 (2017).
[5] Chen, Tianqi, et al. "Xgboost: extreme gradient boosting." R package version 0.4-2 1.4 (2015).
[6] Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.
[7] Cho, Kyunghyun, et al. "On the properties of neural machine translation: Encoder-decoder approaches." arXiv preprint arXiv:1409.1259 (2014).
[8] Vaswani, Ashish, et al. "Attention is all you need." arXiv preprint arXiv:1706.03762 (2017).
[9] Passalis, Nikolaos, et al. "Deep adaptive input normalization for time series forecasting." IEEE transactions on neural networks and learning systems 31.9 (2019): 3760-3765.
[10] Caruana, Rich, et al. "Ensemble selection from libraries of models." Proceedings of the twenty-first international conference on Machine learning. 2004.
