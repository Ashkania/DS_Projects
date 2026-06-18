#!/usr/bin/bash

competition_name="playground-series-s6e6"
file="../output/submission.csv"
message="test"

kaggle competitions submit -c "$competition_name" -f "$file" -m "$message"
echo ''
sleep 20
kaggle competitions submissions -c "$competition_name" | awk 'NR==3 {print $7}'
