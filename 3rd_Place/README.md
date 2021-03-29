# Wind-dependent Variables: Predict Wind Speeds of Tropical Storms
This repo includes code for [Daniel_FG](https://www.drivendata.org/users/Daniel_FG/), 3rd place  
Competition: [Wind-dependent Variables: Predict Wind Speeds of Tropical Storms](https://www.drivendata.org/competitions/72/predict-wind-speeds/page/274/)  


## Solution Summary

The approach is based on two layer pipeline, where the first layer are 14 CNN 
based regression models and the second one is a GBM model adding extra features to the first layer predictions and past predictions. Pre-computed model weights can be downloaded to `Wind_Tropical_Storms/src/models` from `s3://drivendata-competition-radiant-earth/3rd_place/`. A smaller ensemble of 4 models achieves similar performance. See `reports/DrivenData-Competition-Winner-Documentation-3rd.pdf` for additional solution details.

**First layer models**

The first layer is trained over a stratified 4 k-folds split scheme. 

|        | **Desc.**                                                          | **CV RMSE** |
| ------ | ------------------------------------------------------------------ | ----------- |
| A1     | CNN(vgg11) + RNN(gru); SeqLen=8                                    | 7.994       |
| A2     | CNN(vgg11) + RNN(gru); SeqLen=12                                   | 7.726       |
| A3     | CNN(vgg11) + RNN(gru); SeqLen=24                                   | 7.441       |
| A4     | CNN(vgg11) + TRANSFORMERS; SeqLen=24, Gap=0.5 h                    | 7.612       |
| A5     | CNN(vgg11) + TRANSFORMERS; SeqLen=24, Gap=1.0 h                    | 7.516       |
| **A6** | **CNN(vgg11) + RNN(gru); SeqLen=24, Gap=1.0; Predict ALL**         | **7.231**   |
| A7     | CNN(vgg11) + TRANSFORMERS; SeqLen=24, Gap=1.0 h; Predict ALL       | 7.408       |
| B1     | CNN(resnext101\_64x4d)                                             | 8.878       |
| B2     | CNN(se\_resnet152)                                                 | 9.162       |
| B3     | CNN(se\_resnext101\_32x4d)                                         | 9.072       |
| C1     | CNN(se\_resnext101\_32x4d); Channels=3, Gap=1.0 h; SmoothL1 beta=4 | 8.482       |
| C2     | CNN(se\_resnext101\_32x4d); Channels=3, Gap=1.0 h; Diff. Aug.      | 8.476       |
| C3     | CNN(se\_resnext101\_32x4d); Channels=7, Gap=1.0 h; Diff. Aug.      | 8.700       |
| D1     | CNN(vgg16); OpticaFlow vectors; Images=5, Gap=0.5 h                | 12.260      |

(*) See code for more details about the models.

**Second layer models**

This lightGBM model is used to predict wind speed based on 1st level models predictions and 
other features:
- 1st level predictions at time t
- 1st level predictions at time t-1, t-2, t-4, t-6, t-8, t-10, t-12, t-16, t-20, t-24
- Extra features: ocean, storm_duration

The final solution performance is (LocalCV|PublicLB|PrivateLB): **6.4961 | 6.8392 | 6.4561**


The next table shows cross validation scores for 2nd level model depending on the number of 1st level models used:

| **Nb of models** | **Model ID** | **Desc.**                                                          | **CV RMSE** | **%**      |
| ---------------- | ------------ | ------------------------------------------------------------------ | ----------- | ---------- |
| 1                | **A6**       | **CNN(vgg11) + RNN(gru); SeqLen=24, Gap=1.0; Predict ALL**         | 7.0394      | 92.69%     |
| 2                | **A2**       | **CNN(vgg11) + RNN(gru); SeqLen=12**                               | 6.8175      | 95.71%     |
| 3                | **B1**       | **CNN(resnext101\_64x4d)**                                         | 6.7065      | 97.29%     |
| **4**            | **A4**       | **CNN(vgg11) + TRANSFORMERS; SeqLen=24, Gap=0.5 h**                | **6.6537**  | **98.06%** |
| 5                | C1           | CNN(se\_resnext101\_32x4d); Channels=3, Gap=1.0 h; SmoothL1 beta=4 | 6.6149      | 98.64%     |
| 6                | A5           | CNN(vgg11) + TRANSFORMERS; SeqLen=24, Gap=1.0 h                    | 6.5850      | 99.08%     |
| 7                | B2           | CNN(se\_resnet152)                                                 | 6.5707      | 99.30%     |
| 8                | A1           | CNN(vgg11) + RNN(gru); SeqLen=8                                    | 6.5602      | 99.46%     |
| 9                | C3           | CNN(se\_resnext101\_32x4d); Channels=7, Gap=1.0 h; Diff. Aug.      | 6.5500      | 99.61%     |
| 10               | B3           | CNN(se\_resnext101\_32x4d)                                         | 6.5399      | 99.77%     |
| 11               | A7           | CNN(vgg11) + TRANSFORMERS; SeqLen=24, Gap=1.0 h; Predict ALL       | 6.5360      | 99.83%     |
| 12               | A3           | CNN(vgg11) + RNN(gru); SeqLen=24                                   | 6.5296      | 99.92%     |
| 13               | C2           | CNN(se\_resnext101\_32x4d); Channels=3, Gap=1.0 h; Diff. Aug.      | 6.5248      | 100.00%    |
| 14               | D1           | CNN(vgg16); OpticaFlow vectors; Images=5, Gap=0.5 h                | 6.5247 (*)      | 100.00%    |

(*) Final model model score is 6.4961 after averaging 4 of the model’s predictions. See code for more details.

Using just 4 models we can achieve 98% of the score when using all 14 models.


## How to reproduce the solution

### Prerequisites

The following system specification were used for this competition:

- CPU Intel 8 core (16 threads)
- GPU Nvidia RTX 2080Ti 11 GB
- RAM 64 GB
- Ubuntu 16.04

Software prerequisites:

- Python 3.6
- Pipenv
- CUDA 10.1

Clone this repository and  install requirements from project root:

`pipenv install` 

I couldn't install pytorch for cuda 10.1 from pipenv, so needed to install it using pip:

`pipenv run pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

Finally, install pretrained models:

`pipenv run pip install pretrainedmodels==0.7.4`

### Download Data

Download data from https://www.drivendata.org/competitions/72/predict-wind-speeds/data/ and uncompress images to the project directory `data`.

    │
    ├── data
    │    │    
    │    ├── test               <- Test set images.
    │    │    
    │    ├── train              <- Training set images.
    │    │
    │    ├── 4Kfolds_202012220741.csv
    │    │
    │    ├── submission_format.csv
    │    │
    │    ├── test_set_features.csv
    │    │
    │    ├── training_set_features.csv
    │    │
    │    └── training_set_labels.csv

This directory already contains a file `4Kfolds_202012220741.csv` with the 4-folds data split scheme used for 
cross validation and stacking level-1 predictions.  

### Train models and make predictions

If you complied with all the prerequisites just run:

`bash train_predict.sh`

### Run inference using L1 weights

If saved weights of L1 models are already in `models` folder, you can skip the step of 
training those L1 models running:

`bash predict_from_L1weights.sh`

### Final submission

It triggers the training and prediction on the test set. 
When the script has finished running you'll find the prediction in `predictions` folder. 
The final submission would be `L2A_FINAL_submission.csv`.

