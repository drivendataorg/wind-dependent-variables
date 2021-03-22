[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/re-cyclone-home.jpg)


# Wind-dependent Variables: Predict Wind Speeds of Tropical Storms

## Goal of the Competition
Tropical cyclones are one of the costliest natural disasters globally. Hurricanes can cause upwards of 1,000 deaths and $50 billion in damages in a single event, and have been responsible for well over 160,000 deaths in recent history. An accurate diagnostic model for tropical cyclones is critical for disaster preparedness and response.

According to the National Oceanic and Atmospheric Administration (NOAA), storm damage models approximate risk using an exponential or power of wind speed. Today, forecasters estimate the maximum sustained surface wind speed, or intensity, of tropical cyclones using adaptations of a satellite image-based classification method, known as the Dvorak technique. These techniques involve visual inspection of images and are limited by human subjectivity in assessing complex cloud features. There remains a vital need to develop automated, objective, and accurate tropical cyclone intensity estimation tools from readily available satellite image data.

The goal of this challenge is to estimate the wind speeds of storms at different points in time using satellite images captured throughout a storm’s life cycle and the temporal memory of the storm. Radiant Earth Foundation has worked with the NASA IMPACT team to assemble a data set of tropical storm imagery, which includes single-band satellite images at a long-wave infrared frequency and corresponding wind speed annotations. Improving initial wind speed estimates could mean significant improvements in short-term storm intensity forecasting, risk approximation models, and disaster readiness and response.

## What's in this Repository

This repository contains code from winning competitors in the [Wind-dependent Variables: Predict Wind Speeds of Tropical Storms](https://www.drivendata.org/competitions/72/predict-wind-speeds/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1   | [vecxoz](https://www.drivendata.org/users/vecxoz/) | 6.677 | 6.256 | Create 3-channel images by concatenating an image frame with two prior timesteps. Use a TimeDistributed layer to process sequences of 24 images (8 3-channel images). Train 51 models with CNN-1D, CNN-3D, CNN-MEAN, and CNN-LSTM architectures using different backbones and different historical depths. Ensemble predictions using a weighted average optimized by pairwise greedy search over sorted predictions. Backbones include ResNet50, Inception, InceptionResNet, Xception, EfficientNet-B0, EfficientNet-B3, and VisionTransformer-L32. Features are 366 x 366 pixel images with flip and rotation augmentations and corresponding test time augmentations. Classification models using 156 separate classes also contribute to the final ensemble.
2   | [KarimAmer](https://www.drivendata.org/users/KarimAmer/) | 6.867 | 6.417 | Fine-tune pre-trained CNNs, including ResNet50 and ResNet18, on sets of 3-channel images created by concatenating an image frame with two prior timesteps. Sets are composed of 1, 3, 6, 9, or 12 consecutive time steps, or 6 time steps using every other prior image (eg. images 1, 3, 5, 7, 9, 11). Apply test time augmentation to intermediate predictions. Features are 244 x 244 pixel images with flip, rotation, and time augmentations. Train an additional ~200 models with CNN-1D, Xgboost, LSTM, GRU, MLP, Transformer, Decision Tree, and Linear Regression architectures on intermediate predictions as well as 25-30 prior time steps using Deep Adaptive Input Normalization. Ensemble predictions using an ensemble selection algorithm with bagging.
3   | [Daniel_FG](https://www.drivendata.org/users/Daniel_FG/) | 6.839 | 6.456 | Fine-tune 14 models on sequences of 8 or 12 3-channel images with CNN, RNN, GRU, and Transformer architectures, using different backbones and different historical depths. Fine-tune one model using Optical Flow Vectors from a sequence of 5 images spaced 1 hour apart. Backbones include VGG-11, VGG-16, ResNeXt101, SE_Resnet152, and SE_ResNeXt101. Multi-channel images are created from repeating a single channel or concatenating a monochrome image with two prior timesteps of varying depths using a block sampling strategy. Initial features are 244 x 244 pixel images with flip, rotation, and crop augmentations. Train a lightGBM on intermediate predictions, predictions from times t-1 through t-24, ocean, and relative time metadata to output final predictions.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: [How to use Deep Learning to Predict Tropical Storm Wind Speeds](https://www.drivendata.co/blog/predict-wind-speeds-benchmark/)**
