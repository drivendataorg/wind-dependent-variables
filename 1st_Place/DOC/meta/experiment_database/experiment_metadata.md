Metadata for experiment_database.csv
====================================

Fields of experiment database
=============================

experiment_id           Unique experiment identifier corresponding to directory name containing code, weights, and predictions (e.g. 'run-20210105-1225')
LB_private              Private LB score
LB_public               Public LB score
CV_70k                  Cross-validation score computed over the whole training set (70k examples) via 5-fold split (prediction on original images only)
CV_70k_tta9             Cross-validation score computed over the whole training set (70k examples) via 5-fold split (mean of all test-time augmentations)
CV_14k_f0               Hold-out score for fold 0 (14k examples)
CV_14k_f1               Hold-out score for fold 1 (14k examples)
CV_14k_f2               Hold-out score for fold 2 (14k examples)
CV_14k_f3               Hold-out score for fold 3 (14k examples)
CV_14k_f4               Hold-out score for fold 4 (14k examples)
n_epochs_f0             Number of epochs the best model was trained in fact in fold 0 (not greater then `max_n_epochs`)
n_epochs_f1             Number of epochs the best model was trained in fact in fold 1 (not greater then `max_n_epochs`)
n_epochs_f2             Number of epochs the best model was trained in fact in fold 2 (not greater then `max_n_epochs`)
n_epochs_f3             Number of epochs the best model was trained in fact in fold 3 (not greater then `max_n_epochs`)
n_epochs_f4             Number of epochs the best model was trained in fact in fold 4 (not greater then `max_n_epochs`)
ensemble_id             Final ensemble in which model was used ('ens_51_only', or 'ens_22_only', or 'ens_both')
task                    Regression or classification task
architecture            Architecture of the model. Description is given below ('cnn-1ch', 'cnn-3ch-ind', 'cnn-3ch-mean', 'cnn-conv3d', 'cnn-lstm', 'cnn-trans', 'cnn-convlstm2d')
image_resolution        Size of image in pixels (e.g. 366)
image_n_channels        Number of image channels (e.g. 3)
image_inverted          Whether image was inverted (i.e. 255-image) (all 'yes')
n_frames                How many monochrome frames (or channels) are in single example (e.g. 24)
step                    Step used during selection of historical frames (e.g. 1)
historical_depth        Total historical depth. For 'cnn-1ch': 1, for 'cnn-3ch-ind': 2*step+1, for 'cnn-3ch-mean': step*3, for all others: equal to n_frames
backbone                CNN model used as a backbone (e.g. 'ResNet50')
backbone_weights        Name of backbone pretrained weights (e.g. 'imagenet')
batch_size              Batch size (e.g. 64)
learning_rate           Learning rate (e.g. 1e-4)
max_n_epochs            Maximum number of epochs. `n_epochs_f*` cannnot be greater than this value (e.g. 15)
aug_number              Number of training augmentations, i.e. number of variants of image modification during training (all 9)
tta_number              Number of inference augmentations, i.e. number of variants of image modification during inference (all 9)
experiment_description  Short experiment description


Architecture description
========================

'cnn-1ch'
    CNN-Conv2D, 1-channel images (original monochrome)
    Each training example is 3-tensor of shape (366, 366, 1)

'cnn-3ch-ind'
    CNN-Conv2D, 3-channel images, channels are individual monochrome frames taken with some step
    Each training example is 3-tensor of shape (366, 366, 3),
    where one channel is current frame and two others are previous frames with some step
    E.g. channels = [current_minus_18, current_minus_9, current_frame]
    
'cnn-3ch-mean'
    CNN-Conv2D, 3-channel images, channels are means of some ranges of monochrome frames
    Each training example is 3-tensor of shape (366, 366, 3)
    E.g. channels = [
        mean([current_minus_2, current_minus_1, current_frame]),
        mean([current_minus_5, current_minus_4, current_minus_3]),
        mean([current_minus_8, current_minus_7, current_minus_6]),]

'cnn-conv3d'
    CNN-Conv3D, volumes of 3-channel images
    Model which applies Conv3D layers
    Each training example is 4-tensor of shape e.g. (8, 366, 366, 3),    
    where one channel is current frame and all others are consequent previous frames (24 in total)

'cnn-lstm'
    TimeDistributed CNN-Conv2D and LSTM on top
    Each training example is 4-tensor of shape e.g. (8, 366, 366, 3),    
    where one channel is current frame and all others are consequent previous frames (24 in total)

'cnn-trans'
    TimeDistributed CNN-Conv2D and Transformer on top
    Each training example is 4-tensor of shape e.g. (8, 366, 366, 3),    
    where one channel is current frame and all others are consequent previous frames (24 in total)

'cnn-convlstm2d'
    TimeDistributed CNN-Conv2D and ConvLSTM2D on top
    Each training example is 4-tensor of shape e.g. (8, 366, 366, 3),    
    where one channel is current frame and all others are consequent previous frames (24 in total)






