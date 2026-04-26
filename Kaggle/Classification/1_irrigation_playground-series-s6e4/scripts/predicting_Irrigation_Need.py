#!/usr/bin/env python

#TODO:
# 0. Add CV, should edit the best model selection: for no cv and cv, do it inside the
#     train_and_evaluate_models function and return the best model from there.
#     So, the find_best_model function will be only for ensembling methods.
# 1. Add grid serach with the config file, for now only return the best model based on the
#     default hyperparameters.
# 2. Combine predictors using ensemble methods:
#     1. Soft voting, 2. Weighted soft voting, 3. Stacking, 4. Blending(?)
#     Better to define a parameter to select only the best models for ensembling
#     to not include the weak ones.
# 3. Move the simple training with default models into a separate function
# and call it from main.
# 4. Activate SMOTE (and other sampling techniques) in the pipeline and compare results with and without it.
# 5. Plot the results for each model and compare them visually.
# 6. Add more evaluation metrics like ROC-AUC, Precision-Recall curves, etc.
# 7. Add correlation matrix to EDA
# 8. Add feature importance plots for tree-based models.
# 9. train the final model on the entire training data (train + val) before making predictions on test data.

# Hardcoded parameters:
# 1. saved score in models score dict is weighted avg f1-score, but it can be changed to any other metric.
# 2. Columns with ordinal mapping + their order, columns with binary mapping 
# must be defined for each dataset 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json

from argparse import ArgumentParser, RawDescriptionHelpFormatter

from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, StandardScaler, OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    PassiveAggressiveClassifier, Perceptron
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, VotingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

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
            --models all --cv 5
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
        choices=[
            'dt', 'rf', 'lr', 'ridge', 'sgd', 'pa', 'perceptron',
            'et', 'gb', 'hgb', 'xgb', 'lgbm', 'ada', 'svc', 'lsvc',
            'knn', 'gnb', 'qda', 'lda', 'mlp', 'dummy', 'all'
            ],
        help='Models to train: dt (Decision Tree), rf (Random Forest),' \
        'lr (Logistic Regression), ridge (Ridge), sgd (SGD),' \
        'pa (Passive Aggressive), perceptron, et (Extra Trees),' \
        'gb (Gradient Boosting), hgb (HistGradient Boosting),' \
        'xgb (XGBoost), lgbm (LightGBM), ada (AdaBoost), svc (SVC),' \
        'lsvc (Linear SVC), knn (KNN), gnb (Gaussian NB), qda, lda,' \
        'mlp (MLP), dummy, or all (which excludes the slow and weak models like gb, svc, lsvc, qda, and dummy)'
    )
    parser.add_argument(
        '--cv',
        default=False,
        type=int,
        help='Doing cross validation with the specified number of folds (e.g., 5 for 5-fold CV)'
    )
    parser.add_argument(
        '--grid-config-file',
        type=str,
        default=None,
        help='Path to a JSON file containing grid search' \
        'configurations for hyperparameter tuning'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='Path to save the predictions'
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
        X, y, test_size=test_size, random_state=42, stratify=y
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

    return X_train_processed, X_val_processed, y_train_enc, y_val_enc, le, preprocessor

def train_and_evaluate_models(
        X_train_processed, y_train_enc,
        X_val_processed, y_val_enc,
        models, cv=False, grid_config_file=None
    ):
    model_dict = {
        # Linear Models
        'lr': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'ridge': RidgeClassifier(random_state=42, class_weight='balanced'),
        'sgd': SGDClassifier(random_state=42, max_iter=1000, class_weight='balanced'),
        'pa': PassiveAggressiveClassifier(random_state=42, max_iter=1000, class_weight='balanced'),
        'perceptron': Perceptron(random_state=42, max_iter=1000, class_weight='balanced'),
        # Tree-Based Models
        'dt': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'rf': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'et': ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'gb': GradientBoostingClassifier(random_state=42), # ~10 times slower compared to the rest!
        'hgb': HistGradientBoostingClassifier(random_state=42, class_weight='balanced'),
        # Boosting Models
        'xgb': XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
        'lgbm': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced'),
        'ada': AdaBoostClassifier(random_state=42),
        # Support Vector Models
        'svc': SVC(random_state=42), # ~100 times slower compared to the rest!!!
        'lsvc': LinearSVC(random_state=42, max_iter=1000), # Precision ill defined set to 0
        # Instance-Based Models
        'knn': KNeighborsClassifier(n_neighbors=5),
        # Probabilistic Models
        'gnb': GaussianNB(),
        'qda': QuadraticDiscriminantAnalysis(reg_param=0.1), # Add regularization to handle singular covariance matrices
        'lda': LinearDiscriminantAnalysis(),
        # Neural Networks
        'mlp': MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
        # Specialized Models
        'dummy': DummyClassifier(strategy='most_frequent'), # Precision ill defined set to 0
    }
    if models == ['all']:
        models = list(model_dict.keys())
        # Exclude slow and weak models from 'all'
        exclude_models = ['gb', 'svc', 'lsvc', 'qda', 'dummy']
        models = [model for model in models if model not in exclude_models]

    if grid_config_file:
        pass
        # Load grid search configurations from the specified JSON file
        # with open(grid_config_file, 'r') as f:
        #     grid_configs = json.load(f)
        # # Update model_dict with the specified hyperparameters for grid search
        # for model_name, params in grid_configs.items():
        #     if model_name in model_dict:
        #         model_dict[model_name] = GridSearchCV(
        #             estimator=model_dict[model_name],
        #             param_grid=params,
        #             cv=cv if cv else 5,
        #             n_jobs=-1,
        #             verbose=1
        #         )
        #     else:
        #         print(f"Warning: Model '{model_name}' not found in model_dict. Skipping grid search for this model.")
        #### for now, return only the best model here too.
    elif cv:
        model_scores = {}
        
        for model_name in models:
            
            start = time.time()
            
            print(f"\n{'='*50}")
            print(f"Training {model_name.upper()}...")
            print('='*50)
            
            model = model_dict[model_name]

            cv_results = cross_validate(
                model, X_train_processed, y_train_enc,
                cv=cv, scoring=['f1_macro', 'f1_weighted'], n_jobs=-1
            )

            mean_score = cv_results['test_f1_macro'].mean()
            model_scores[model_name] = mean_score

            end = time.time()

            print(f"Time taken: {end - start:.2f} seconds")
            print(f"Cross-Validation Results for {model_name.upper()}:")
            print(f"Mean Macro F1-Score: {mean_score:.4f}")

        # first find the best model based on cv scores, then fit it on the entire training data (train + val) before making predictions on test data
        # Fit once on full training data for later use in test prediction
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = model_dict[best_model_name]

        # train on the entire training data (train + val) before making predictions on test data.
        best_model.fit(
            np.concatenate([X_train_processed, X_val_processed]),
            np.concatenate([y_train_enc, y_val_enc])
        )

        print(f"\n{'='*50}")
        print(f"Best model: {best_model_name.upper()} with CV Macro F1-Score: {model_scores[best_model_name]:.4f}")
        print('='*50)

    else:
        model_scores = {}
        
        for model_name in models:
            
            start = time.time()
            print(f"\n{'='*50}")
            print(f"Training {model_name.upper()}...")
            print('='*50)

            model = model_dict[model_name]

            model.fit(X_train_processed, y_train_enc)
            y_pred = model.predict(X_val_processed)
            report = classification_report(y_val_enc, y_pred, output_dict=True, digits=3)
            model_scores[model_name] = report['macro avg']['f1-score']

            end = time.time()
            
            print(f"Time taken: {end - start:.2f} seconds")
            print(f"\nClassification Report for {model_name.upper()}:")
            print(classification_report(y_val_enc, y_pred, digits=3))
            print(f"\nConfusion Matrix for {model_name.upper()}:")
            print(confusion_matrix(y_val_enc, y_pred))

        best_model_name = max(model_scores, key=model_scores.get)
        best_model = model_dict[best_model_name]

        # train on the entire training data (train + val) before making predictions on test data.
        best_model.fit(
            np.concatenate([X_train_processed, X_val_processed]),
            np.concatenate([y_train_enc, y_val_enc])
        )

        print(f"\n{'='*50}")
        print(f"Best model: {best_model_name.upper()} with Validation Macro F1-Score: {model_scores[best_model_name]:.4f}")
        print('='*50)


    return best_model 
    
def combine_models(trained_models, model_scores, combination_method=None):
    """
    Find the best model based on the provided model scores and combination method.
    This is meant to be only for the grid search case, for the cv and non cv cases,
    the best model is already selected in the train_and_evaluate_models function
    and returned from there.
    """
    # model_scores = {'model_name': weighted_avg_f1_score, ...}
    if combination_method == 'soft_voting':
        # Implement soft voting logic here (e.g., average probabilities)
        # return voting_model
        pass
    elif combination_method == 'weighted_soft_voting':
        # Implement weighted soft voting logic here (e.g., weighted average of probabilities)
        # return weighted_voting_model
        pass
    elif combination_method == 'stacking':
        # Implement stacking logic here (e.g., train a meta-model on the predictions of base models)
        # return stacked_model
        pass
    elif combination_method == 'blending':
        # Implement blending logic here (e.g., train a meta-model on a holdout set)
        # return blended_model
        pass
    # else:
    #     best_model_name = max(model_scores, key=model_scores.get)
    #     print(f"\n{'='*50}")
    #     print(f"Best model: {best_model_name.upper()} with: {model_scores[best_model_name]:.4f}")
    #     print('='*50)
    #     best_model = trained_models[best_model_name]
    #     return best_model

def predict_test_data(test, preprocessor, model, le):
    # Apply the same preprocessing to test data
    test_processed = preprocessor.transform(test)
    print(f"Processed test shape: {test_processed.shape}")

    ####
    # first fit the model on the entire training data (train + val) before making predictions on test data.
    ####
    # model.fit(X_train_processed, y_train_enc)

    # Make predictions
    y_pred = model.predict(test_processed)

    # Inverse transform to get original labels
    y_pred_labels = le.inverse_transform(y_pred)

    print(f"\nTest predictions: {y_pred_labels}")
    print(f"Unique predicted labels: {np.unique(y_pred_labels)}")
    print(f"\nNumber of predictions: {len(y_pred_labels)}")

    return y_pred_labels

def main():
    args = command_line_args()
    train_path = args.train_dataset
    test_path = args.test_dataset
    test_size = args.test_size
    drop_columns = args.drop_columns
    models = args.models
    output_name = args.output
    cv=args.cv
    grid_config_file=args.grid_config_file

    train, test, test_id = load_data(
        train_path, test_path, drop_columns
    )
    target_variable = args.target_variable or train.columns[-1]

    if args.eda:
        eda(train, target_variable)

    (
        X_train_processed,
        X_val_processed,
        y_train_enc,
        y_val_enc,
        le,
        preprocessor
    ) = preprocess_data(train, target_variable, test_size)

    # before:
    # trained_models, model_scores = train_and_evaluate_models(
    #     X_train_processed, y_train_enc, X_val_processed, y_val_enc, models, cv, grid_config_file
    # )

    # now for the cv and non cv cases, the best model is already selected in the train_and_evaluate_models function
    
    best_model = train_and_evaluate_models(
        X_train_processed, y_train_enc, X_val_processed, y_val_enc, models, cv, grid_config_file
    )


    # best_model = find_best_model(trained_models,model_scores)

    test_predictions = predict_test_data(test, preprocessor, best_model, le)
    

    # Save predictions to a file or return them
    output = pd.DataFrame({'id': test_id, 'Irrigation_Need': test_predictions})
    output.to_csv(output_name, index=False)
    print(f"Predictions saved to {output_name}")

    print('Done!')

if __name__ == '__main__':
    main()