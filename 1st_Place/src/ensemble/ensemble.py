#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
"""
Usage:
python3 ensemble.py \
    --data_dir=../../data \
    --model_dir=../../models \
    --out_dir=./ \
    --ens_id=51 \
"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../../data', help='Data directory')
parser.add_argument('--model_dir', type=str, default='../../models', help='Model/predictions directory')
parser.add_argument('--out_dir', type=str, default='./', help='Output directory')
parser.add_argument('--ens_id', type=int, default=51, choices=[51, 22], help='Ensemble id (51 or 22)')
args = parser.parse_args()
print('Command line arguments:')
print(args)

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

n_folds = 5
n_tta = 9 # 0: orig image only, 9: orig images + 9 tta (10 preds total)
csv_name = 'submission_ens_%d.csv' % args.ens_id

train_df = pd.read_csv(os.path.join(args.data_dir, 'train_cv.csv'))
subm_df = pd.read_csv(os.path.join(args.data_dir, 'submission_format.csv'))

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Specify model dirs

ens_51_dirs = [
'run-20210126-0329', 'run-20210125-1843', 'run-20210124-0112', 'run-20210121-1931', 'run-20210126-2217',
'run-20210122-1753', 'run-20210122-1801', 'run-20210126-2313', 'run-20210125-1412', 'run-20210121-2121',
'run-20210107-0150', 'run-20210104-2211', 'run-20210104-2209', 'run-20210107-0152', 'run-20210110-0122',
'run-20210110-2151', 'run-20210107-0151', 'run-20210110-2148', 'run-20210107-0149', 'run-20210110-0109',
'run-20210118-1734', 'run-20210108-1903', 'run-20210124-0156', 'run-20210124-0119', 'run-20210120-0012',
'run-20210120-1324', 'run-20210106-0050', 'run-20210106-0047', 'run-20210124-0131', 'run-20210121-1912',
'run-20210106-0051', 'run-20210107-1527', 'run-20210105-1443', 'run-20210126-1835', 'run-20210104-2157',
'run-20210107-1532', 'run-20210126-1905', 'run-20210112-0019', 'run-20210127-1320', 'run-20210107-1530',
'run-20210110-2200', 'run-20210128-1354', 'run-20210128-0011', 'run-20210128-0014', 'run-20210128-0016',
'run-20210128-0047', 'run-20210128-1545', 'run-20210128-1940', 'run-20210128-1341', 'run-20210128-0045',
'run-20210129-1511',
]

ens_22_dirs = [
'run-20210126-0329', 'run-20210125-1843', 'run-20210124-0112', 'run-20210121-1931', 'run-20210126-2217',
'run-20210122-1753', 'run-20210122-1801', 'run-20210126-2313', 'run-20210125-1412', 'run-20210121-2121',
'run-20210107-0150', 'run-20210104-2211', 'run-20210107-0152', 'run-20210110-0122', 'run-20210107-0151',
'run-20210118-1734', 'run-20210106-0047', 'run-20210112-0019', 'run-20210110-2200',  'run-20210131-1822',
'run-20210131-1840', 'run-20210131-1951'
]

if args.ens_id == 51:
    dirs = ens_51_dirs
else:
    dirs = ens_22_dirs

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Ensembling %d models' % len(dirs))

print('Collecting VAL predictions...')
y_preds = []
for counter, d in enumerate(dirs):
    all_tta = []
    for tta_id in range((n_tta + 1)):
        all_folds = []
        for fold_id in range(n_folds):
            all_folds.append(np.load(os.path.join(args.model_dir, d, 'preds', 'y_pred_val_fold_%d_tta_%d.npy' % (fold_id, tta_id))).ravel())
        all_tta.append(np.hstack(all_folds))
    y_preds.extend(all_tta)
    if counter % 10 == 0:
        print(counter)
assert len(y_preds) == (n_tta + 1) * len(dirs)


# Collect coresponding true label
y_true_list = []
for fold_id in range(n_folds):
    y_true_list.append(train_df.loc[train_df['fold_id'] == fold_id, 'wind_speed'].values.ravel())
y_true = np.hstack(y_true_list)


# Compute scores
scores = []
for y_pred in y_preds:
    y_pred = np.int32(np.round(y_pred))
    score = mean_squared_error(y_true, y_pred, squared=False)
    scores.append(score)


# Sort based on scores
# !! ASCENDING (best scores first)
# !! Each prediction in a ROW
sorting_ids = np.argsort(scores)
y_preds_sorted = np.array(y_preds)[sorting_ids]


# Define metric
def metric(y_true, y_pred, rounded=False):    
    if rounded:
        y_pred = np.int32(np.round(y_pred))
    score = mean_squared_error(y_true, y_pred, squared=False)
    return score

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Optimizing ensemble coefs...')

# Range of coefs
coef_range = np.arange(0, 1.11, 0.01)

coefs_best = []
pred_best = y_preds_sorted[0]
score_best = 100

for pred_id in range(1, (n_tta + 1) * len(dirs)):
    coef_best = 1
    for coef in coef_range:
        y_pred = coef * pred_best + (1 - coef) * y_preds_sorted[pred_id]
        score_current = metric(y_true, y_pred, rounded=False)
        if score_current < score_best:
            score_best = score_current
            coef_best = coef
    coefs_best.append(coef_best)
    pred_best = coef_best * pred_best + (1 - coef_best) * y_preds_sorted[pred_id]

assert len(coefs_best) == (n_tta + 1) * len(dirs) - 1
# print('BEST score RAW:         %.6f' % score_best)

# CHECK: apply best coefs to VAL
y_pred_final = y_preds_sorted[0]
for pred_id in range(1, (n_tta + 1) * len(dirs)):
    y_pred_final = coefs_best[pred_id-1] * y_pred_final + (1 - coefs_best[pred_id-1]) * y_preds_sorted[pred_id]
# print('BEST score RAW (check): %.6f' % metric(y_true, y_pred_final, rounded=False))
print('BEST score ROUNDED:     %.6f' % metric(y_true, y_pred_final, rounded=True))

#------------------------------------------------------------------------------
# Create submission for test set
#------------------------------------------------------------------------------

print('Collecting TEST predictions...')
y_preds_test = []
for counter, d in enumerate(dirs):
    for tta_id in range((n_tta + 1)):
        y_preds_folds = []
        for fold_id in range(n_folds):
            y_preds_folds.append( np.load(os.path.join(args.model_dir, d, 'preds', 'y_pred_test_fold_%d_tta_%d.npy' % (fold_id, tta_id))).ravel() )
        y_preds_test.append(np.mean(y_preds_folds, axis=0))
    if counter % 10 == 0:
        print(counter)
assert len(y_preds_test) == (n_tta + 1) * len(dirs)

# Sort according to val predictions order
# !! Each prediction in a ROW
y_preds_test_sorted = np.array(y_preds_test)[sorting_ids]

# Apply best coefs
y_pred_test_final = y_preds_test_sorted[0]
for pred_id in range(1, (n_tta + 1) * len(dirs)):
    y_pred_test_final = coefs_best[pred_id-1] * y_pred_test_final + (1 - coefs_best[pred_id-1]) * y_preds_test_sorted[pred_id]

y_pred_test_final = np.int32(np.round(y_pred_test_final))

print('Creating submission...')

subm_df['wind_speed'] = y_pred_test_final
subm_df.to_csv(os.path.join(args.out_dir, csv_name), index=False)
subm_df.head()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

