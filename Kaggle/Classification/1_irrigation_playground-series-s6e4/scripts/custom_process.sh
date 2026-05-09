#!/usr/bin/bash

models="xgb"

# step 1:
# feed custome process with train:
# it adds a new target with values low, medhigh
./custom_process.py \
    --dataset ../data/train.csv \
    --target Irrigation_Need \
    --output ../data/train_with_binary.csv \
    --step 1


# feed main script with train_new_1.csv:
# it predicts low, medhigh for test.csv and save it in test_new.csv
./predicting_Irrigation_Need.py \
    --train-dataset ../data/train_with_binary.csv \
    --test-dataset ../data/test.csv \
    --drop-columns id Irrigation_Need \
    --target-variable 'binary_target' \
    --output ../data/test_low_vs_rest.csv \
    --models ${models} \
    --grid-config-file ../data/grid_config.cfg
    # --models-to-combine ${models} \

# step 2a:
# feed custom process with train_new_1:
# It toss out low rows in train_new_1:
./custom_process.py \
    --dataset ../data/train_with_binary.csv \
    --target Irrigation_Need \
    --output ../data/train_subset.csv \
    --step 2

##########################################
# # step 2b:
# # feed custome process with test.csv:3
# # It toss out low rows, then drop prediction col:
# ./custom_process.py \
#     --dataset ../data/test.csv \
#     --target Irrigation_Need \
#     --output ../data/test_subset.csv \
#     --step 2
# #### does not having a clear column in test a problem?
##################################################


# feed main script with train_new_2.csv:
# it predict medium, high:
./predicting_Irrigation_Need.py \
    --train-dataset ../data/train_subset.csv \
    --test-dataset ../data/test.csv \
    --drop-columns id binary_target \
    --target-variable Irrigation_Need \
    --output ../data/test_med_high.csv \
    --models ${models} \
    --grid-config-file ../data/grid_config.cfg
    # --models-to-combine ${models} \

# # step 3:
# # Simple join 2 test dataset:
# # test_low_vs_rest.csv, test_med_high.csv
./custom_process.py \
    --dataset ../data/test_low_vs_rest.csv \
    --dataset-additional ../data/test_med_high.csv \
    --output ../data/submission.csv \
    --step 3
