#!/usr/bin/bash

competition_name="playground-series-s6e4"
file="../output/submission5.csv"
message="test"

kaggle competitions submit -c "$competition_name" -f "$file" -m "$message"
echo ''
sleep 20
kaggle competitions submissions -c "$competition_name" | awk 'NR==3 {print $7}'
