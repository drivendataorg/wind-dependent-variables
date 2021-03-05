# Instructions

DOC.tar.gz is full solution except model weights (.h5). Size is about 800 MB. It contains all code and predictions (.npy). So after downloading this archive it is possible to run all actions except inference. Model weights are in 6 separate archives. Total size is about 65 GB. Largest part is about 18 GB.

So first we download DOC.tar.gz and extract it. Then rename “models_copy_without_weights” directory just to align with code from instructions:

curl -L -o DOC.tar.gz https://www.dropbox.com/s/aq3o491mqp8ns4i/DOC.tar.gz?dl=0
tar xzf DOC.tar.gz
cd DOC
mv models_copy_without_weights models

We are on clean Ubuntu 18.04.
We need to replace "requirements.txt" with a new version and install Python packages:

cd ~/DOC/src
curl -L -o requirements.txt https://www.dropbox.com/s/syj8nwyum4r64fg/requirements.txt?dl=0
sudo apt-get update
sudo apt-get -y install python3-pip
pip3 install --upgrade pip
pip3 install -r requirements.txt

Now we need to download competition CSV files. At the current stage we don't need images.
(The following code is included in "DOC/src/data/run_data_creation_in_parallel.sh".)

cd ~/DOC/data
curl -L -o submission_format.csv <link>
curl -L -o training_set_features.csv <link>
curl -L -o training_set_labels.csv <link>
curl -L -o test_set_features.csv <link>

Finally let's create CSV files containing training labels and describing CV split (i.e. fold ids):

cd ~/DOC/src/data
python3 create_cv_split.py --data_dir=../../data

Now we are ready to compute ensembles and any model scores.

cd ~/DOC/src/ensemble
python3 ensemble.py --ens_id=51

At the current point we can run all steps which do not require model weights i.e. compute ensemble submission, compute CV scores for each model, create data, retrain models.

DOC/meta/STEPS_TO_REPRODUCE.sh – all instructions to reproduce solution. This is Section 2
DOC/meta/ WRITEUP.doc – solution write up i.e. answers to questions from Section 3
DOC/meta/experiment_database – organized results and parameters of my experiments
DOC/meta/experiment_database/experiment_database.ipynb – usage example

To run inference we need model weights. Before downloading the weights we need to rename “models” directory and create new empty “models” directory, then download 6 parts in “models” directory and extract. Nothing needs to be copied from old “models” directory.

cd DOC
mv models models_copy_without_weights
mkdir models
cd models
curl -L -o models_part_1.tar.gz https://www.dropbox.com/s/sxqzrbpasyq4zee/models_part_1.tar.gz?dl=0
curl -L -o models_part_2.tar.gz https://www.dropbox.com/s/9oygxzbdbbghzol/models_part_2.tar.gz?dl=0
curl -L -o models_part_3.tar.gz https://www.dropbox.com/s/c04pim7dnh3nqta/models_part_3.tar.gz?dl=0
curl -L -o models_part_4.tar.gz https://www.dropbox.com/s/64i3lqrrahbm5i7/models_part_4.tar.gz?dl=0
curl -L -o models_part_5.tar.gz https://www.dropbox.com/s/u10o1f55bd8occ3/models_part_5.tar.gz?dl=0
curl -L -o models_part_6.tar.gz https://www.dropbox.com/s/9ar614n85dq04mu/models_part_6.tar.gz?dl=0
tar xzf models_part_*.tar.gz

