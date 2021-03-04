#!/bin/sh

echo "Test Inference from first stage CNN models"

# ResNet50 1 time step image
echo "ResNet50 1 time step image"
python3 conv_inference.py #output folder -> resnet50_models

# ResNet50 with 3 time step images
echo "ResNet50 with 3 time step images"
python3 prepare_dataset_v1.py
python3 3ts_conv_inference.py #output folder -> 3ts_imgs_resnet50_models_10folds

# ResNet50 with 6 time step images
echo "ResNet50 with 6 time step images"
python3 prepare_dataset_v2.py --time_steps 6
python3 6ts_conv_inference.py --ch_arrangement 0 #output folder -> 6ts_imgs_time_aug_resnet50_models
python3 6ts_conv_inference.py --ch_arrangement 1 #output folder -> 6ts_imgs_time_aug_resnet50_models_021
python3 6ts_conv_inference.py --ch_arrangement 2 #output folder -> 6ts_imgs_time_aug_resnet50_models_102
python3 6ts_conv_inference.py --ch_arrangement 3 #output folder -> 6ts_imgs_time_aug_resnet50_models_120
python3 6ts_conv_inference.py --ch_arrangement 4 #output folder -> 6ts_imgs_time_aug_resnet50_models_201
python3 6ts_conv_inference.py --ch_arrangement 5 #output folder -> 6ts_imgs_time_aug_resnet50_models_210

# ResNet50 with 9 time steps images
echo "ResNet50 with 9 time step images"
python3 prepare_dataset_v2.py --time_steps 9
python3 9ts_conv_inference.py #output folder -> 9ts_imgs_time_aug_resnet50_models

# ResNet18 with 12 time steps images
echo "ResNet18 with 12 time steps images"
python3 prepare_dataset_v2.py --time_steps 12
python3 12ts_conv_inference.py #output folder -> 12ts_imgs_resnet18_models

# ResNet50 with 6 time steps images, 2 space
echo "ResNet50 with 6 time steps images, 2 space"
python3 prepare_dataset_v3.py --step_size 2
python3 6ts_2space_conv_inference.py #output folder -> 6ts_2space_imgs_time_aug_resnet50_models

# ResNet50 with 6 time steps images, 3 space
echo "ResNet50 with 6 time steps images, 3 space"
python3 prepare_dataset_v3.py --step_size 3
python3 6ts_3space_conv_inference.py #output folder -> 6ts_3space_imgs_time_aug_resnet50_models

# ResNet50 with 6 time steps images, 4 space
echo "ResNet50 with 6 time steps images, 4 space"
python3 prepare_dataset_v3.py --step_size 4
python3 6ts_4space_conv_inference.py #output folder -> 6ts_4space_imgs_time_aug_resnet50_models

# ResNet50 with 6 time steps images, 5 space
echo "ResNet50 with 6 time steps images, 5 space"
python3 prepare_dataset_v3.py --step_size 5
python3 6ts_5space_conv_inference.py #output folder -> 6ts_5space_imgs_time_aug_resnet50_models

# ResNet50 with 6 time steps images, 6 space
echo "ResNet50 with 6 time steps images, 6 space"
python3 prepare_dataset_v3.py --step_size 6
python3 6ts_6space_conv_inference.py #output folder -> 6ts_6space_imgs_time_aug_resnet50_models

echo "Training second stage stacking models"

# CNN
echo "CNN"
python3 ts_cnn.py --input_cnn_idx 0
python3 ts_cnn.py --input_cnn_idx 1
python3 ts_cnn.py --input_cnn_idx 2
python3 ts_cnn.py --input_cnn_idx 3
python3 ts_cnn.py --input_cnn_idx 4
python3 ts_cnn.py --input_cnn_idx 5
python3 ts_cnn.py --input_cnn_idx 6
python3 ts_cnn.py --input_cnn_idx 7
python3 ts_cnn.py --input_cnn_idx 8
python3 ts_cnn.py --input_cnn_idx 9
python3 ts_cnn.py --input_cnn_idx 10
python3 ts_cnn.py --input_cnn_idx 11
python3 ts_cnn.py --input_cnn_idx 12
python3 ts_cnn.py --input_cnn_idx 13
python3 ts_cnn.py --input_cnn_idx 14

# MLP
echo "MLP"
python3 mlp.py --input_cnn_idx 0
python3 mlp.py --input_cnn_idx 1
python3 mlp.py --input_cnn_idx 2
python3 mlp.py --input_cnn_idx 3
python3 mlp.py --input_cnn_idx 4
python3 mlp.py --input_cnn_idx 5
python3 mlp.py --input_cnn_idx 6
python3 mlp.py --input_cnn_idx 7
python3 mlp.py --input_cnn_idx 8
python3 mlp.py --input_cnn_idx 9
python3 mlp.py --input_cnn_idx 10
python3 mlp.py --input_cnn_idx 11
python3 mlp.py --input_cnn_idx 12
python3 mlp.py --input_cnn_idx 13
python3 mlp.py --input_cnn_idx 14

# LSTM
echo "LSTM"
python3 lstm.py --input_cnn_idx 0
python3 lstm.py --input_cnn_idx 1
python3 lstm.py --input_cnn_idx 2
python3 lstm.py --input_cnn_idx 3
python3 lstm.py --input_cnn_idx 4
python3 lstm.py --input_cnn_idx 5
python3 lstm.py --input_cnn_idx 6
python3 lstm.py --input_cnn_idx 7
python3 lstm.py --input_cnn_idx 8
python3 lstm.py --input_cnn_idx 9
python3 lstm.py --input_cnn_idx 10
python3 lstm.py --input_cnn_idx 11
python3 lstm.py --input_cnn_idx 12
python3 lstm.py --input_cnn_idx 13
python3 lstm.py --input_cnn_idx 14

# GRU
echo "GRU"
python3 gru.py --input_cnn_idx 0
python3 gru.py --input_cnn_idx 1
python3 gru.py --input_cnn_idx 2
python3 gru.py --input_cnn_idx 3
python3 gru.py --input_cnn_idx 4
python3 gru.py --input_cnn_idx 5
python3 gru.py --input_cnn_idx 6
python3 gru.py --input_cnn_idx 7
python3 gru.py --input_cnn_idx 8
python3 gru.py --input_cnn_idx 9
python3 gru.py --input_cnn_idx 10
python3 gru.py --input_cnn_idx 11
python3 gru.py --input_cnn_idx 12
python3 gru.py --input_cnn_idx 13
python3 gru.py --input_cnn_idx 14

# Transformer
echo "Transformer"
python3 ts_transformer.py --input_cnn_idx 0
python3 ts_transformer.py --input_cnn_idx 1
python3 ts_transformer.py --input_cnn_idx 2
python3 ts_transformer.py --input_cnn_idx 3
python3 ts_transformer.py --input_cnn_idx 4
python3 ts_transformer.py --input_cnn_idx 5
python3 ts_transformer.py --input_cnn_idx 6
python3 ts_transformer.py --input_cnn_idx 7
python3 ts_transformer.py --input_cnn_idx 8
python3 ts_transformer.py --input_cnn_idx 9
python3 ts_transformer.py --input_cnn_idx 10
python3 ts_transformer.py --input_cnn_idx 11
python3 ts_transformer.py --input_cnn_idx 12
python3 ts_transformer.py --input_cnn_idx 13
python3 ts_transformer.py --input_cnn_idx 14

# Linear Regression
echo "Linear Regression"
python3 linear_reg.py --input_cnn_idx 0
python3 linear_reg.py --input_cnn_idx 1
python3 linear_reg.py --input_cnn_idx 2
python3 linear_reg.py --input_cnn_idx 3
python3 linear_reg.py --input_cnn_idx 4
python3 linear_reg.py --input_cnn_idx 5
python3 linear_reg.py --input_cnn_idx 6
python3 linear_reg.py --input_cnn_idx 7
python3 linear_reg.py --input_cnn_idx 8
python3 linear_reg.py --input_cnn_idx 9
python3 linear_reg.py --input_cnn_idx 10
python3 linear_reg.py --input_cnn_idx 11
python3 linear_reg.py --input_cnn_idx 12
python3 linear_reg.py --input_cnn_idx 13
python3 linear_reg.py --input_cnn_idx 14

# Decision Tree
echo "Decision Tree"
python3 dt.py --input_cnn_idx 0
python3 dt.py --input_cnn_idx 1
python3 dt.py --input_cnn_idx 2
python3 dt.py --input_cnn_idx 3
python3 dt.py --input_cnn_idx 4
python3 dt.py --input_cnn_idx 5
python3 dt.py --input_cnn_idx 6
python3 dt.py --input_cnn_idx 7
python3 dt.py --input_cnn_idx 8
python3 dt.py --input_cnn_idx 9
python3 dt.py --input_cnn_idx 10
python3 dt.py --input_cnn_idx 11
python3 dt.py --input_cnn_idx 12
python3 dt.py --input_cnn_idx 13
python3 dt.py --input_cnn_idx 14

# Xgboost
echo "Xgboost"
python3 xgb_Bulldozer.py

# Third Stage Model
echo "Third Stage Model"
python3 es_ensemble.py
