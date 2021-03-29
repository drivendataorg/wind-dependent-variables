#!/usr/bin/env bash

echo "################ 1. Pre process Optical Flow images ###################################"
pipenv run python src/preprocess_optical_flow.py

echo "#######################################################################################"
echo


echo "################ 2. predict first level CNN models ##############################"

pipenv run python src/predict_L1A_models.py --model_id A1
pipenv run python src/predict_L1A_models.py --model_id A2
pipenv run python src/predict_L1A_models.py --model_id A3
pipenv run python src/predict_L1A_models.py --model_id A4
pipenv run python src/predict_L1A_models.py --model_id A5
pipenv run python src/predict_L1A_models.py --model_id A6
pipenv run python src/predict_L1A_models.py --model_id A7

pipenv run python src/predict_L1A_models.py --model_id B1
pipenv run python src/predict_L1A_models.py --model_id B2
pipenv run python src/predict_L1A_models.py --model_id B3

pipenv run python src/predict_L1A_models.py --model_id C1
pipenv run python src/predict_L1A_models.py --model_id C2
pipenv run python src/predict_L1A_models.py --model_id C3

pipenv run python src/predict_L1A_models.py --model_id D1

echo "################ 2.1 Concatenate L1 predictions #######################################"

pipenv run python src/concatenate_L1A_predictions.py --model_id A1 
pipenv run python src/concatenate_L1A_predictions.py --model_id A2 
pipenv run python src/concatenate_L1A_predictions.py --model_id A3 
pipenv run python src/concatenate_L1A_predictions.py --model_id A4 
pipenv run python src/concatenate_L1A_predictions.py --model_id A5 
pipenv run python src/concatenate_L1A_predictions.py --model_id A6 
pipenv run python src/concatenate_L1A_predictions.py --model_id A7 

pipenv run python src/concatenate_L1A_predictions.py --model_id B1 
pipenv run python src/concatenate_L1A_predictions.py --model_id B2 
pipenv run python src/concatenate_L1A_predictions.py --model_id B3 

pipenv run python src/concatenate_L1A_predictions.py --model_id C1 
pipenv run python src/concatenate_L1A_predictions.py --model_id C2 
pipenv run python src/concatenate_L1A_predictions.py --model_id C3

pipenv run python src/concatenate_L1A_predictions.py --model_id D1

echo "#######################################################################################"
echo


echo "################ 3. Train/predict second level models #################################"

pipenv run python src/L2A_FINAL_lightgbm.py

echo "#######################################################################################"
echo
