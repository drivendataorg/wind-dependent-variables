# Introduction

Commands and their explanations are listed below. Pre-computed model weights corresponding to the best epoch in each fold (not greater than max value specified in settings in each "run.py" file) as well as intermediate predictions for val and test (.npy) can be downloaded to `DOC/models` from a private s3 bucket. To request access, please [contact us](https://www.drivendata.org/contact/). 3rd party weights can be downloaded to `DOC/weights_3rd_party` from the link provided in "INFO.md".

The best submission (LB 6.2558) is an ensemble of 51 models. A smaller ensemble of 22 models achieves similar performance (LB 6.3046). These models are specified in `ensemble_v2.py`. See `reports/DrivenData-Competition-Winner-Documentation-1st.pdf` for additional solution details. Practical tips for using subsets of the winning model are outlined in `DOC/meta/STEPS_TO_REPRODUCE.sh`.

# Directory structure

```
LICENSE.txt
ensemble_v2.py
run_data_creation_in_sequence.sh
run_inference_8x_V100.sh
run_inference_TPUv3_8.sh
DOC/
    meta/
        experiment_database/
            experiment_database.csv
            experiment_database.ipynb
            experiment_metadata.md
        STEPS_TO_REPRODUCE.sh
    models/
        run-20210104-2157/
            run.py
        run-20210131-1951/
        ...
    src/
        data/
            ...
        ensemble/
            ensemble.py
        requirements.txt
        vecxoz_utils.py
    weights_3rd_party/
        INFO.md
```

# Getting started

1. Installing dependencies, downloading weights, downloading and processing data:

Ubuntu 18.04 with preinstalled CUDA 11.0 and cuDNN 8.0.4.
2 cores, 8 GB RAM, 1 TB SSD
Expected runtime: about 20 hours

```
cd $HOME
bash run_data_creation_in_sequence.sh
```

2. Run inference:

32 cores, 120 GB RAM, 1 TB SSD, 8x V100 GPU
Expected runtime on full test set (44k): about 100 hours

```
cd $HOME
bash run_inference_8x_V100.sh
```

3. Run ensemble:

Expected runtime: < 1 min

```
cd $HOME
python3 ensemble_v2.py \
--data_dir=$HOME/DOC/data \
--model_dir=$HOME/DOC/models \
--out_dir=./ \
--ens_id=51
```


# Additional instructions

```
"DOC/models" dir contains 54 subdirs corresponding to each individual
    model (experiment) included in final solution.
    51 of 54 - the best private score
    22 of 54 - smaller ensemble which also reached 1st place
        (except additional 3 models it is intersected with 51)
All 54 supplied models follow exactly the same experiment design (all scores are directly comparable):
    5 folds (split is based on storm id)
    10 predictions from each fold for val and 10 predictions for test
    10 predictions are: 1 original image and 9 test-time augmentations:
        h-flip, v-flip, and rotations multiple of 45 degrees
As a result each experiment folder contains:
    single code file "run.py" (code structure is the same for all experiments)
Each "run.py" is a self-contained experiment file to run all jobs (--help option is supported):
        train
        predict val set
        compute val scores
        predict test set
        create submission .csv
    Each file 'run.py' follows the same structure and has same command line arguments.
    The most important parts is model definition ("init_model" function) and
    data reading ("parse_example" function).
Term "single model" means self-ensemble of all folds and all TTA.
Model file name description (e.g. model-best-f0-e014-8.7136.h5)
    f0 - fold id (from 0 to 4)
    e014 - number of epochs trained in fact (mostly not greater than 15 or 20)
    8.7136 - score for given fold computed with predictions on original val images (no TTA)
Ensemble computation
    Ensemble is formulated as weighted arithmetic average
    optimized using pair-wise greedy search over sorted predictions.
    Concept is the following:
        Sort predictions from all models based on CV score
            in descending order. Smallest (i.e. best) value is last.
            Predictions from each TTA are sorted independently.
        Find best coefficients for pair of 1st and 2nd prediction, then apply these coefficients and compute result
        Find best coefficients for pair of previous result and 3rd prediction, and so on
Data creation
    Script "create_cv_split.py" creates validation split i.e. it creates file "train_cv.csv" with "fold_id" column.
        All my experiments were run in the same setup using same split, so all scores are directly comparable.
        If organizers are interested to compare some of their own models with mine they can retrain them using this split.
    Script "run_data_creation_in_parallel.sh" has section "Loading original competition data"
        where expired links must be replaced with valid ones
    There are 15 datasets total
    Each dataset creation is performed in 2 stages:
        compile examples with previous frames and save on disk
        then write these examples in TFRecord files split by fold
    There are 4 different scripts corresponding to 4 different concepts of TFREcord example creation:
        each example is 1-channel image (original)
        each example is 3-channel image, one channel is current frame and 2 others are historical frames taken with some step
        each example is 3-channel image, each channel is mean of some range of historical frames
        each example is a list of some number of 1-channel image (original), i.e. current and several previous historical frames
    Each of 4 scripts has 'Compliance check' section in code. If uncommented
        these print statements show what files (previous frames) are used in fact to create given example
    Parallel processing was tested on machine with
        16 cores, 16 GB RAM and 1 TB free space (required) on HDD and took about 5 hours to complete.
        SSD disk should speed up this process.
Training parameters
    All training parameters (batch_size, learning_rate) are optimized for TPUv3-8.
    For most of the models these parameters will hold on 8x V100 GPU with mixed precision.
    For a single GPU batch size and learning rate should be reduced proportionally.
    Also gradient accumulation can be used (not implemented in my solution).
    Some training batch size values for reference:
                                                TPUv3-8     P100
    CNN, EfficientNet-B7                        192         12
    CNN-LSTM, 24 frames, EfficientNet-B3        64          4
Experiment database
    DOC/meta/experiment_database contains experiment statistics
```
