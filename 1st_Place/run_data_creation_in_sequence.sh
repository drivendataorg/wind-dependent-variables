#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

echo "***** Download base archive (everything without model weights)..."
cd $HOME
curl -L -o DOC.tar.gz https://www.dropbox.com/s/6sig3tclqrc161y/DOCv2.tar.gz?dl=0
tar xzf DOC.tar.gz
cd $HOME/DOC
mkdir models

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

echo "***** Install requirements..."
cd $HOME/DOC/src
sudo apt-get update
sudo apt-get -y install libgl1-mesa-glx # just in case (for cv2)
sudo apt-get -y install python3-pip
pip3 install --upgrade pip
pip3 install -r requirements.txt

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

echo "***** IMPORTANT: Please insert correct csv file links"
echo "***** Download and extract all competition data (csv and images)..."
cd $HOME/DOC/data
curl -L -o submission_format.csv "INSERT LINK"
curl -L -o training_set_features.csv "INSERT LINK"
curl -L -o training_set_labels.csv "INSERT LINK"
curl -L -o test_set_features.csv "INSERT LINK"
curl -L -o re-train-images.tgz https://drivendata-public-assets-eu.s3.eu-central-1.amazonaws.com/re-train-images.tgz
curl -L -o re-test-images.tgz https://drivendata-public-assets-eu.s3.eu-central-1.amazonaws.com/re-test-images.tgz
tar xzf re-train-images.tgz
tar xzf re-test-images.tgz

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

echo "***** Download and extract all model weights..."
cd $HOME/DOC/models
curl -L -o models_part_1.tar.gz https://www.dropbox.com/s/sxqzrbpasyq4zee/models_part_1.tar.gz?dl=0
curl -L -o models_part_2.tar.gz https://www.dropbox.com/s/9oygxzbdbbghzol/models_part_2.tar.gz?dl=0
curl -L -o models_part_3.tar.gz https://www.dropbox.com/s/c04pim7dnh3nqta/models_part_3.tar.gz?dl=0
curl -L -o models_part_4.tar.gz https://www.dropbox.com/s/64i3lqrrahbm5i7/models_part_4.tar.gz?dl=0
curl -L -o models_part_5.tar.gz https://www.dropbox.com/s/u10o1f55bd8occ3/models_part_5.tar.gz?dl=0
curl -L -o models_part_6.tar.gz https://www.dropbox.com/s/9ar614n85dq04mu/models_part_6.tar.gz?dl=0

tar xzf models_part_1.tar.gz
tar xzf models_part_2.tar.gz
tar xzf models_part_3.tar.gz
tar xzf models_part_4.tar.gz
tar xzf models_part_5.tar.gz
tar xzf models_part_6.tar.gz

rm models_part_1.tar.gz
rm models_part_2.tar.gz
rm models_part_3.tar.gz
rm models_part_4.tar.gz
rm models_part_5.tar.gz
rm models_part_6.tar.gz

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

echo "***** Start data processing..."
echo "***** Create CV split..."

cd $HOME/DOC/src/data
DATA_DIR=$HOME/DOC/data
python3 create_cv_split.py --data_dir=$DATA_DIR

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

echo "***** Create TFRecords..."

python3 create_data_for_cnn_1d.py --data_dir=$DATA_DIR

python3 create_data_for_cnn_ind.py --data_dir=$DATA_DIR --step=5
python3 create_data_for_cnn_ind.py --data_dir=$DATA_DIR --step=8
python3 create_data_for_cnn_ind.py --data_dir=$DATA_DIR --step=9
python3 create_data_for_cnn_ind.py --data_dir=$DATA_DIR --step=10
python3 create_data_for_cnn_ind.py --data_dir=$DATA_DIR --step=11
python3 create_data_for_cnn_ind.py --data_dir=$DATA_DIR --step=12
python3 create_data_for_cnn_ind.py --data_dir=$DATA_DIR --step=15

python3 create_data_for_cnn_mean.py --data_dir=$DATA_DIR --step=3
python3 create_data_for_cnn_mean.py --data_dir=$DATA_DIR --step=10
python3 create_data_for_cnn_mean.py --data_dir=$DATA_DIR --step=15

python3 create_data_for_cnn_lstm.py --data_dir=$DATA_DIR --n_frames=24 --step=1
python3 create_data_for_cnn_lstm.py --data_dir=$DATA_DIR --n_frames=48 --step=1
python3 create_data_for_cnn_lstm.py --data_dir=$DATA_DIR --n_frames=60 --step=1
python3 create_data_for_cnn_lstm.py --data_dir=$DATA_DIR --n_frames=96 --step=1

echo "***** DONE"

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



