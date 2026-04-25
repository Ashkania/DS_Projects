#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser, RawDescriptionHelpFormatter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report, confusion_matrix

def command_line_args():
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description="""
            Predicting Irrigation Need:
            This script loads the training and test datasets,
            performs exploratory data analysis (EDA) if specified,
            and prepares the data for modeling.
            Usage:
            ./predicting_Irrigation_Need.py --train-dataset ../data/train.csv
            --test-dataset ../data/test.csv --drop-columns id --eda
            """
        )
    parser.add_argument(
        '--train-dataset',
        type=str,
        required=True,
        help='Path to the training dataset'
    )
    parser.add_argument(
        '--test-dataset',
        type=str,
        required=True,
        help='Path to the test dataset'
    )
    parser.add_argument(
        '--drop-columns',
        nargs='+',
        default=[],
        help='List of columns to drop from the dataset'
    )
    parser.add_argument(
        '--target-variable',
        type=str,
        default=None,
        help='Name of the target variable in the dataset'
    )
    parser.add_argument(
        '--eda',
        action='store_true',
        help='Perform exploratory data analysis'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of the dataset to include in the test split'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        choices=['dt', 'rf', 'lr', 'all'],
        help='Models to train: dt (Decision Tree), rf (Random Forest), lr (Logistic Regression), or all'
    )
    return parser.parse_args()

def load_data(train_path, test_path, drop_columns=[]):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test_id = test['id']

    for col in drop_columns:
        print(f'Dropping column: {col}')
        for data in [train, test]:
            data.drop(columns=[col], inplace=True)

    # train: col1, col2, ..., target
    # test: col1, col2, ...
    # test_id: col_id
    return train, test, test_id

def eda(train_data, target_variable):

    num_cols = train_data.select_dtypes(include=np.number).columns
    cat_cols = train_data.select_dtypes(include='object').columns

    print('Performing Exploratory Data Analysis...')
    print(f'\nColumns:\n{train_data.columns}')
    print(f'\nSample Data:\n{train_data.sample(5)}')
    print(f'\nInfo:\n{train_data.info()}')
    print(f'\nDescription:\n{train_data.describe()}')
    if train_data.size < 10000:  # Only plot pairplot for smaller datasets to avoid performance issues
        sns.pairplot(train_data, hue=target_variable)
        plt.show()
    print(f'\nMissing Values:\n{train_data.isnull().sum()}')
    print(f'\nDuplicates:\n{train_data.duplicated().sum()}')

    for i, col in enumerate(cat_cols):
        print(f'\nUnique Values for {col}:\n{train_data[col].unique()}')

    # Figure1:
    print(f'\nClass Distribution:\n{train_data[target_variable].value_counts()}')
    plt.figure(figsize=(10, 4))
    plt.pie(
        train_data[target_variable].value_counts(),
        labels=train_data[target_variable].value_counts().index,
        autopct='%1.1f%%'
    )
    plt.title(f"Distribution of Labels")

    # Figure2:
    print('\nBox Plots for Numerical Features:')
    _, axes1 = plt.subplots(4, 3, figsize=(14, 10))  # Reduce height to make boxes smaller
    axes1 = axes1.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(train_data[col], ax=axes1[i], orient='h')
        axes1[i].set_title(f"Distribution of {col}")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout()

    # print('\nCount Plots for Categorical Features:')
    # _, axes2 = plt.subplots(3, 3, figsize=(12, 8))
    # axes2 = axes2.flatten()
    # for i, col in enumerate(cat_cols):
    #     sns.histplot(train_data[col], ax=axes2[i])
    #     axes2[i].set_title(f"Distribution of {col}")
    #     axes2[i].tick_params(axis='x', rotation=45)

    plt.show()

    print('EDA Completed.')

def preprocess_data(train, target_variable, test_size):
    X = train.drop(columns=[target_variable])
    y = train[target_variable]
    # X: col1, col2, ...
    # y: target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    # X_train: col1, col2, ..., y_train: target
    # X_val: col1, col2, ..., y_val: target

    # Encode target variable
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    # Define column types
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()

    # Define ordinal columns and their mappings
    ordinal_cols = ['Soil_Type', 'Crop_Growth_Stage']
    Soil_Type_mapping = ['Sandy', 'Loamy', 'Silt', 'Clay']
    Crop_Growth_Stage_mapping = ['Sowing', 'Vegetative', 'Flowering', 'Harvest']
    ordinal_mappings = [Soil_Type_mapping, Crop_Growth_Stage_mapping]

    # Define binary columns and their mappings
    binary_cols = ['Mulching_Used']
    Mulching_Used_mapping = ['No', 'Yes']

    # One hot encoding for remaining categorical columns
    onehot_cols = [col for col in cat_cols if col not in ordinal_cols + binary_cols]

    # Create preprocessor with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(categories=ordinal_mappings), ordinal_cols),
            ('binary', OrdinalEncoder(categories=[Mulching_Used_mapping]), binary_cols),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), onehot_cols),
            ('num', StandardScaler(), num_cols)
        ],
        remainder='passthrough'
    )

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Processed validation shape: {X_val_processed.shape}")

    return X_train_processed, X_val_processed, y_train_enc, y_val_enc, le

def train_and_evaluate_models(X_train, X_val, y_train, y_val, models):
    # Define model mappings
    model_dict = {
        'dt': DecisionTreeClassifier(random_state=42),
        'rf': RandomForestClassifier(random_state=42, n_jobs=-1),
        'lr': LogisticRegression(random_state=42, max_iter=1000)
    }

    # If 'all' is selected, use all models
    if 'all' in models:
        models = ['dt', 'rf', 'lr']

    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}...")
        print('='*50)

        model = model_dict[model_name]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        print(f"\nClassification Report for {model_name.upper()}:")
        print(classification_report(y_val, y_pred))

        print(f"\nConfusion Matrix for {model_name.upper()}:")
        print(confusion_matrix(y_val, y_pred))

def main():
    args = command_line_args()
    train_path = args.train_dataset
    test_path = args.test_dataset
    test_size = args.test_size
    drop_columns = args.drop_columns
    models = args.models

    train, test, test_id = load_data(train_path, test_path, drop_columns)
    target_variable = args.target_variable or train.columns[-1]

    if args.eda:
        eda(train, target_variable)

    X_train_processed, X_val_processed, y_train_enc, y_val_enc, le = preprocess_data(train, target_variable, test_size)

    train_and_evaluate_models(X_train_processed, X_val_processed, y_train_enc, y_val_enc, models)
        
    print('Done!')

if __name__ == '__main__':
    main()