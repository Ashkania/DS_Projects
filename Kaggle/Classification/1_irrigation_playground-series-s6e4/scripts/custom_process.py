#!/usr/bin/env python

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import pandas as pd
import numpy as np

# get the train file, add new target column
# run the main processor 

def command_line_args():
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description="""
            ...
        """
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the dataset'
    )
    parser.add_argument(
        '--dataset-additional',
        type=str,
        required=False,
        help='Path to the dataset 2'
    )
    parser.add_argument(
        '--step',
        type=int,
        required=True,
        help='step number'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Name of the target variable in the dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save the predictions'
    )

    return parser.parse_args()


def step_one(dataset_path, target):

    train = pd.read_csv(dataset_path)
    train['binary_target'] = 'Medhigh'
    train.loc[
        (train[target] == 'Low'), 'binary_target'
        ] = 'Low'
    return train
    
def step_two(dataset_path, target):
    df = pd.read_csv(dataset_path)
    sub_df = df.loc[
        df[target] != 'Low'
        ]
    return sub_df

def step_three(df1_path, df2_path):
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    merged = df1.merge(
        df2, on='id', suffixes=('_df1', '_df2')
    )
    merged['Irrigation_Need'] = np.where(
        merged['Irrigation_Need_df1'] == 'Low',
        'Low',
        merged['Irrigation_Need_df2']
    )

    merged = merged[['id', 'Irrigation_Need']]

    return merged

def main():
    args = command_line_args()
    dataset = args.dataset
    dataset2 = args.dataset_additional
    step = args.step
    target = args.target
    output = args.output

    if step == 1:
        train_with_binary = step_one(dataset, target)
        train_with_binary.to_csv(output, index=False)
        
    elif step == 2:
        sub_df = step_two(dataset, target)
        sub_df.to_csv(output, index=False)

    elif step == 3:
        result = step_three(dataset, dataset2)
        result.to_csv(output, index=False)

if __name__ == '__main__':
    main()
