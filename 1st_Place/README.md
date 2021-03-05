# =========================
# INSTRUCTIONS to REPRODUCE
# =========================
# 
# 1st place solution
# "Wind-dependent Variables: Predict Wind Speeds of Tropical Storms" competition
# https://www.drivendata.org/competitions/72/predict-wind-speeds/
# Author: Igor Ivanov (vecxoz@gmail.com)
# MIT license
# 
# 
# Introduction
# ============
# 
# Solution package contains not only model weights but also all predictions
# so it is possible to compute all scores and prediction stats (e.g. correlation)
# without training and inference. Also there is an experiment database
# where I collected scores and experiment parameters. 
# Commands and their explanation are listed in section "STEPS" below
# 
# "DOC/models" dir contains 54 subdirs corresponding to each individual 
#     model (experiment) included in final solution.
#     51 of 54 - the best private score
#     22 of 54 - smaller ensemble which also reached 1st place 
#         (except additional 3 models it is intersected with 51)
# All 54 supplied models follow exactly the same experiment design (all scores are directly comparable):
#     5 folds (split is based on storm id)
#     10 predictions from each fold for val and 10 predictions for test 
#     10 predictions are: 1 original image and 9 test-time augmentations:
#         h-flip, v-flip, and rotations multiple of 45 degrees
# As a result each experiment folder contains:
#     single code file "run.py" (code structure is the same for all experiments)
#     5 model weights files corresponding to the best epoch in each fold
#         Best epoch may be different across folds, but not greater than max value     
#         specified in settings in each "run.py" file (mostly 15 or 20).
#     "preds" dir with 100 .npy files containing predictions for val and test
# Each "run.py" is a self-contained experiment file to run all jobs (--help option is supported):
#         train
#         predict val set
#         compute val scores
#         predict test set
#         create submission .csv
#     Each file 'run.py' follows the same structure and has same command line arguments.
#     The most important parts is model definition ("init_model" function) and 
#     data reading ("parse_example" function).
# Term "single model" means self-ensemble of all folds and all TTA.
# Model file name description (e.g. model-best-f0-e014-8.7136.h5)
#     f0 - fold id (from 0 to 4)
#     e014 - number of epochs trained in fact (mostly not greater than 15 or 20)
#     8.7136 - score for given fold computed with predictions on original val images (no TTA)
# Ensemble computation
#     Ensemble is formulated as weighted arithmetic average
#     optimized using pair-wise greedy search over sorted predictions.
#     Concept is the following:
#         Sort predictions from all models based on CV score 
#             in descending order. Smallest (i.e. best) value is last.
#             Predictions from each TTA are sorted independently.
#         Find best coefficients for pair of 1st and 2nd prediction, then apply these coefficients and compute result
#         Find best coefficients for pair of previous result and 3rd prediction, and so on
# Data creation
#     Script "create_cv_split.py" creates validation split i.e. it creates file "train_cv.csv" with "fold_id" column.
#         All my experiments were run in the same setup using same split, so all scores are directly comparable.
#         If organizers are interested to compare some of their own models with mine they can retrain them using this split.
#     Script "run_data_creation_in_parallel.sh" has section "Loading original competition data"
#         where expired links must be replaced with valid ones
#     There are 15 datasets total
#     Each dataset creation is performed in 2 stages:
#         compile examples with previous frames and save on disk
#         then write these examples in TFRecord files split by fold
#     There are 4 different scripts corresponding to 4 different concepts of TFREcord example creation:
#         each example is 1-channel image (original)
#         each example is 3-channel image, one channel is current frame and 2 others are historical frames taken with some step
#         each example is 3-channel image, each channel is mean of some range of historical frames
#         each example is a list of some number of 1-channel image (original), i.e. current and several previous historical frames
#     Each of 4 scripts has 'Compliance check' section in code. If uncommented 
#         these print statements show what files (previous frames) are used in fact to create given example
#     Parallel processing was tested on machine with 
#         16 cores, 16 GB RAM and 1 TB free space (required) on HDD and took about 5 hours to complete. 
#         SSD disk should speed up this process.
# Training parameters
#     All training parameters (batch_size, learning_rate) are optimized for TPUv3-8.
#     For most of the models these parameters will hold on 8x V100 GPU with mixed precision.
#     For a single GPU batch size and learning rate should be reduced proportionally.
#     Also gradient accumulation can be used (not implemented in my solution).
#     Some training batch size values for reference:
#                                                 TPUv3-8     P100
#     CNN, EfficientNet-B7                        192         12
#     CNN-LSTM, 24 frames, EfficientNet-B3        64          4
# Experiment database
#     DOC/meta/experiment_database contains experiment statistics
# 
# Directory structure and description
# ===================================
# 
# DOC/
#     LICENSE.txt
#     data/
#     meta/
#         experiment_database/
#         STEPS_TO_REPRODUCE.sh
#         WRITEUP.doc
#     models/
#         run-20210104-2157/
#             preds/
#                 y_pred_val_fold_0_tta_0.npy
#                 ...
#                 y_pred_test_fold_4_tta_9.npy
#                 y_pred_test_fold_0_tta_0.npy
#                 ...
#                 y_pred_val_fold_4_tta_9.npy
#             model-best-f0-e014-8.7136.h5
#             ...
#             model-best-f4-e014-8.5117.h5
#             run.py
#         ...
#         run-20210131-1951/
#     models_copy_without_weights/
#         <same structure as "models" but without weights>
#     src/
#         data/
#             ...
#         ensemble/
#             ensemble.py
#         requirements.txt
#         vecxoz_utils.py
#     weights_3rd_party/
#     
# 
# DOC/data 
#     In this dir all original data should be downloaded and all processed data 
#     will be created in corresponding subdirs
# 
# DOC/meta/experiment_database
#     Contains experiment statistics in .csv file
#     Example notebook
#     Metadata describing fields of experiment database and architectures
#
# DOC/meta/STEPS_TO_REPRODUCE.sh
#     This file.
#     All instructions
# 
# DOC/meta/WRITEUP.doc
#     Solution writeup
#
# DOC/src/data
#     Code to create all datasets
# 
# DOC/src/ensemble
#     Code to create ensemble from all predictions (.npy)
# 
# DOC/src/vecxoz_utils.py
#     Single utility library from which all scripts get import
# 
# DOC/models
#     !! If after extraction this dir is empty than we need to download parts of archive
#       and extract here
#     Contains 54 subdirs corresponding to each individual model
#     e.g. "run-20210104-2157"
# 
# DOC/models/run-20210104-2157/preds
#     Contains 100 .npy files: predictions for val and test (5 folds x 10)
# 
# DOC/models/run-20210104-2157/model-best-f0-e014-8.7136.h5
#     Model weights
# 
# DOC/models/run-20210104-2157/run.py
#     Experiment code
# 
# DOC/models_copy_without_weights
#     This is complete copy (code and predictions) of "DOC/models" dir but without weights
#     In case of large data this dir is present to give quick access to code and predictions
#         without need to download large file
# 
# DOC/weights_3rd_party
#     Pretrained weights for C3D model
# 
# 
#
# =============================================================================
# STEPS
# =============================================================================
# 
# 
# =============================================================================
# 1. Reproduce ensemble from saved predictions (.npy). Runtime: 1 min
# =============================================================================

cd DOC/src/ensemble

# Create ensemble from 51 models corresponding to the best private LB 6.2558
# Result "submission_ens_51.csv" will be saved in current dir

python3 ensemble.py --ens_id=51

# Create smaller ensemble from 22 models (still 1st place, private LB 6.3046)

python3 ensemble.py --ens_id=22


# =============================================================================
# 2. Compute val scores and create submission from saved predictions (.npy) 
#    for any specific model. Runtime: <1 min
# =============================================================================

# Let's take one of the best single models which by itself can reach 7th place in private LB 6.6216
# This is CNN-LSTM, 48 frames, EfficientNet-B0 backbone

cd DOC/models/run-20210122-1753

# All folds, all TTA
# Submission .csv is saved in the current dir

python3 run.py \
    --job=score_subm \
    --data_preds_dir=preds \
    --tta_number=9 \

# All folds, without TTA (i.e. score for original test images only)

python3 run.py \
    --job=score \
    --data_preds_dir=preds \
    --tta_number=0 \

# Look at the score of 1st single fold without TTA
# This score is close to the one mentioned in weights file name (e.g. "model-best-f0-e014-7.0310.h5")

python3 run.py \
    --job=score \
    --data_preds_dir=preds \
    --tta_number=0 \
    --n_folds=1


# Let's look at the scores of the overall best model based on CV (have no LB score)
# This is CNN-LSTM, 24 frames, ResNet50 backbone

cd DOC/models/run-20210126-0329

python3 run.py \
    --job=score_subm \
    --data_preds_dir=preds \
    --tta_number=9 \

# =============================================================================
# 3. Create datasets
# =============================================================================

# **********************
# 
# !! We must place correct links at the top of "run_data_creation_in_parallel.sh" script
# !! 1 TB of free space is required to create data in parallel
#    eventually size of all data will be around 500 GB
#
#***********************

# Script runs creation of all datasets in parallel (each process is started inside detached screen)
# Runtime: 5 hours on a machine with 16 cores, 16 GB RAM

cd DOC/src/data
bash run_data_creation_in_parallel.sh

# We can also create single specific dataset for a given model
# To find out what script to run we have to look inside run.py file of corresponding model
# E.g. let's open run-20210121-1931/run.py and look at the value of argument "--data_tfrec_dir"
# this is '../../data/data-tfrec-frames024-step001-lstm'
# By the suffix "lstm" we can identify that we need script "create_data_for_cnn_lstm.py"
# and also we have 24 frames and step 1, so the command will be:

cd DOC/src/data
python3 create_data_for_cnn_lstm.py \
    --data_dir=../../data \
    --n_frames=24 \
    --step=1

# =============================================================================
# 4. Run inference using saved weights (.h5) for any specific model
# =============================================================================

# Before we begin with inference we need to have model weights
# If DOC/models dir is empty than we need to download and extract weights inside DOC/models

# Let's take another good model reaching 8th place in private LB (6.6325)
# This is CNN-LSTM, 24 frames, EfficientNet-B3 backbone
# Runtime for validation only inference (14k examples), batch 32, P100 GPU, 8 cores, 30 GB RAM: 
#   6 min per 1 TTA, 
#   5 hours total (5 folds * 10 TTA * 6 min)
# Runtime on TPU, batch 128:
#   35 sec per 1 TTA
#   29 min total (5 folds * 10 TTA * 35 sec)

cd DOC/models/run-20210121-1931

# Predict validation set and compute scores
# 'tpu_ip_or_name': None for GPU, or node name ('node-1') for TPU
# 'mixed_precision': 'mixed_float16' applicable to GPU, None means default system precision (probably 'float32')

python3 run.py \
    --job=val_score \
    --tpu_ip_or_name=None \
    --mixed_precision=mixed_float16 \
    --data_preds_dir=preds_recreated \
    --batch_size=32 \
    --tta_number=9 \
    --use_cache=False \


# =============================================================================
# 5. Run trainig and inference for any specific model
# =============================================================================

# This is CNN-LSTM, 24 frames, EfficientNet-B3 backbone
# Runtime for training, batch 4, P100 GPU, 8 cores, 30 GB RAM: 
#   2 hours per epoch
#   150 hours total (5 folds * 15 epochs * 2 hours)
# Runtime on TPU, batch 64
#   20 min per epoch
#   25 hours total (5 folds * 15 epochs * 20 min)

cd DOC/models/run-20210121-1931

# Move supplied weights to some dir
mkdir weights_01
mv *.h5 weights_01

python3 run.py \
    --job=train_val_test_score_subm \
    --tpu_ip_or_name=None \
    --mixed_precision=mixed_float16 \
    --data_preds_dir=preds_recreated_from_retrained \
    --batch_size=4 \
    --tta_number=9 \
    --use_cache=False \
    --buffer_size=64 \


# =============================================================================
# =============================================================================


