#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# We are inside `DOC/src/data`
# We will download .csv and .jpg in `DOC/data` (`../../data`)
# All TFRecord dirs will be subdirs of `DOC/data`

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

DATA_DIR=../../data

echo "Loading original competition data..."

curl -o $DATA_DIR/submission_format.csv "https://drivendata-prod.s3.amazonaws.com/data/72/public/submission_format.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYVI2LMPSY%2F20210208%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210208T104816Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=41ce975b2c007ce9d32716a0a116e7e2ffc40c412c548d32c1179cfd35f4657d" ; \
curl -o $DATA_DIR/training_set_features.csv "https://drivendata-prod.s3.amazonaws.com/data/72/public/training_set_features.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYVI2LMPSY%2F20210208%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210208T104816Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e7e57a875f44c8f3373bdcbe2f0863e37fdbf28110e5aa744c416bc22edd72d8" ; \
curl -o $DATA_DIR/training_set_labels.csv "https://drivendata-prod.s3.amazonaws.com/data/72/public/training_set_labels.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYVI2LMPSY%2F20210208%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210208T104816Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e72b3dfbee9493525448d6c633abf609f69a1031414550ac6192c5938e8b98da" ; \
curl -o $DATA_DIR/test_set_features.csv "https://drivendata-prod.s3.amazonaws.com/data/72/public/test_set_features.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYVI2LMPSY%2F20210208%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210208T104816Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=77d8c6b4203248ba8c8b27bd4741df83ea80b14539c7d70ea27de114d5d2b063" ; \
curl -o $DATA_DIR/re-train-images.tgz https://drivendata-public-assets-eu.s3.eu-central-1.amazonaws.com/re-train-images.tgz ; \
curl -o $DATA_DIR/re-test-images.tgz https://drivendata-public-assets-eu.s3.eu-central-1.amazonaws.com/re-test-images.tgz ; \
tar xzf $DATA_DIR/re-train-images.tgz -C $DATA_DIR ; \
tar xzf $DATA_DIR/re-test-images.tgz -C $DATA_DIR ; \
rm $DATA_DIR/re-train-images.tgz ; \
rm $DATA_DIR/re-test-images.tgz ; \

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

echo "create_cv_split.py"
python3 create_cv_split.py --data_dir=$DATA_DIR

#------------------------------------------------------------------------------
# CNN-LSTM
#------------------------------------------------------------------------------

echo "create_data_for_cnn_lstm.py"
for var in 24 48 60 96
do
    screen -d -m python3 create_data_for_cnn_lstm.py --data_dir=$DATA_DIR --n_frames=$var --step=1
done

#------------------------------------------------------------------------------
# CNN-PRE
#------------------------------------------------------------------------------

echo "create_data_for_cnn_ind.py"
for var in 5 8 9 10 11 12 15
do
    screen -d -m python3 create_data_for_cnn_ind.py --data_dir=$DATA_DIR --step=$var
done

#------------------------------------------------------------------------------
# CNN-MEAN
#------------------------------------------------------------------------------

echo "create_data_for_cnn_mean.py"
for var in 3 10 15
do
    screen -d -m python3 create_data_for_cnn_mean.py --data_dir=$DATA_DIR --step=$var
done

#------------------------------------------------------------------------------
# CNN-1D
#------------------------------------------------------------------------------

echo "create_data_for_cnn_1d.py"
screen -d -m python3 create_data_for_cnn_1d.py --data_dir=$DATA_DIR

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


