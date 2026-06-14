#!/usr/bin/env python

#TODO:
# 0. Add CV, should edit the best model selection: for no cv and cv, do it inside the
#     train_and_evaluate_models function and return the best model from there.
#     So, the find_best_model function will be only for ensembling methods.
# 5. Plot the results for each model and compare them visually.
# 6. Add more evaluation metrics like ROC-AUC, Precision-Recall curves, etc.
# 7. Add correlation matrix to EDA
# 8. Add feature importance plots for tree-based models.

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
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    PassiveAggressiveClassifier, Perceptron
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix

class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    """A minimal sklearn-compatible PyTorch classifier wrapper.

    This trains a simple feed-forward network with one hidden layer and
    exposes `fit`, `predict` and `predict_proba` so it can be used alongside
    other sklearn estimators in the script.
    """
    def __init__(
            self,
            input_dim=None,
            hidden_dim=64,
            epochs=20,
            batch_size=32,
            lr=1e-3,
            verbose=False,
            device=None
            ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self._model = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_model(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        n_samples, n_features = X.shape
        n_classes = int(np.max(y)) + 1

        if self.input_dim is None:
            self.input_dim = n_features

        self._model = self._build_model(self.input_dim, n_classes).to(self.device)
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                out = self._model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / n_samples:.4f}")

        return self

    def predict_proba(self, X):
        if torch is None:
            raise RuntimeError('PyTorch is not available. Install torch to use TorchNNClassifier')
        X = np.asarray(X, dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self.device)
            out = self._model(xb)
            probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


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
            --test-dataset ../data/test.csv [--drop-columns id] [--eda]
            [--models all] [--cv 5] [--imbalance smote]
            [--grid-config-file ../data/grid_config.cfg]
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
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of the dataset to include in the test split'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=[
            'dt', 'rf', 'lr', 'ridge', 'sgd', 'pa', 'perceptron',
                'et', 'gb', 'hgb', 'xgb', 'lgbm', 'ada', 'svc', 'lsvc',
                'knn', 'gnb', 'qda', 'lda', 'mlp', 'dummy', 'cat',
                'rf_ovr', 'torch', 'all'
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
        '--imbalance',
        default=False,
        choices=['smote','other']
    )
    parser.add_argument(
        '--grid-config-file',
        type=str,
        default=None,
        help='Path to a JSON file containing grid search' \
        'configurations for hyperparameter tuning'
    )
    parser.add_argument(
        '--models-to-combine',
        nargs='+',
        default=None,
        choices=[
            'dt', 'rf', 'lr', 'ridge', 'sgd', 'pa', 'perceptron',
            'et', 'gb', 'hgb', 'xgb', 'lgbm', 'ada', 'svc', 'lsvc',
            'knn', 'gnb', 'qda', 'lda', 'mlp', 'dummy', 'cat', 'torch', 'all'
            ]
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../output/submission.csv',
        help='Path to save the predictions'
    )
    return parser.parse_args()




def load_data(train_path, test_path, drop_columns=[]):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test_id = test['id']

    for col in drop_columns:
        try:
            print(f'Dropping column: {col}')
            for data in [train, test]:
                data.drop(columns=[col], inplace=True)
        except:
            print(f'Could not delete column: {col} in train or test')

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

    return (
        X_train_processed, X_val_processed,
        y_train_enc, y_val_enc,
        le, preprocessor
    )


def get_resampler(imbalance_type):

    if imbalance_type == 'smote':
            return SMOTE(random_state=42)
    elif imbalance_type == 'under':
        return RandomUnderSampler(random_state=42)
    return None


def get_model_dict(models_to_combine=None):
    """Stores base models and dynamically creates ensembles if requested."""
    model_dict = {
        # Linear Models
        'lr': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'ridge': RidgeClassifier(random_state=42, class_weight='balanced'),
        'sgd': SGDClassifier(random_state=42, max_iter=1000, class_weight='balanced', loss='log_loss'),
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
        'cat' : CatBoostClassifier(random_state=42, verbose=0),
        # Simple PyTorch model (sklearn-compatible wrapper)
        'torch': TorchNNClassifier(),
        # One vs Rest
        'rf_ovr': OneVsRestClassifier(
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        ),
    }

    # Dynamically build ensembles if the user provided a list of models
    if models_to_combine:
        # Create the [('name', model_obj)] list required by Sklearn
        # We use clone() so the ensemble gets fresh instances, not shared memory
        estimators = [
            (name, clone(model_dict[name])) 
            for name in models_to_combine if name in model_dict
        ]
        
        if estimators:
            model_dict['custom_voting'] = VotingClassifier(
                estimators=estimators, voting='soft', n_jobs=-1  # if error for models without a probability: have to use hard voting
            )
            model_dict['custom_stacking'] = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(class_weight='balanced'),
                n_jobs=-1
            )
            
    return model_dict


def build_pipeline(model, imbalance_type=False):
    """Wraps the model in an imblearn pipeline if necessary."""
    if imbalance_type:
        resampler = get_resampler(imbalance_type) # Assuming you have this helper function
        return ImbPipeline([('resampler', resampler), ('model', model)])
    return model

def run_grid_search(
        X_train, y_train,
        models, model_dict,
        grid_config_file,
        cv, imbalance
    ):
    """Handles Grid Search CV logic."""
    with open(grid_config_file, 'r') as f:
        grid_configs = json.load(f)

    model_scores, best_estimators = {}, {}

    for model_name in models:
        if model_name not in model_dict:
            continue
            
        base_model = model_dict[model_name]
        params = grid_configs.get(model_name, {})
        estimator_to_use = build_pipeline(base_model, imbalance)

        # Prepend 'model__' to params if using a Pipeline
        pipeline_params = {f'model__{k}': v for k, v in params.items()} if imbalance else params

        print(f"\n{'='*50}\nTraining {model_name.upper()} with GridSearchCV...\n{'='*50}")
        start = time.time()
        
        grid_search = GridSearchCV(
            estimator=estimator_to_use, param_grid=pipeline_params,
            cv=cv if cv else 5, scoring='f1_macro', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_score = grid_search.best_score_
        model_scores[model_name] = best_score
        best_estimators[model_name] = grid_search.best_estimator_

        print(f"Best params: {grid_search.best_params_}")
        print(f"Time taken: {time.time() - start:.2f} seconds")
        print(f"Best CV Macro F1-Score: {best_score:.4f}")

    best_name = max(model_scores, key=model_scores.get)
    return best_name, best_estimators[best_name], model_scores[best_name]

def run_cv(
        X_train, y_train,
        models, model_dict,
        cv, imbalance
    ):
    """Handles standard Cross Validation logic."""
    model_scores, best_pipelines = {}, {}

    for model_name in models:
        print(f"\n{'='*50}\nTraining {model_name.upper()} with CV...\n{'='*50}")
        start = time.time()
        
        estimator_to_use = build_pipeline(model_dict[model_name], imbalance)
        
        cv_results = cross_validate(
            estimator_to_use, X_train, y_train,
            cv=cv, scoring=['f1_macro', 'f1_weighted'], n_jobs=-1
        )

        mean_score = cv_results['test_f1_macro'].mean()
        model_scores[model_name] = mean_score
        best_pipelines[model_name] = estimator_to_use

        print(f"Time taken: {time.time() - start:.2f} seconds")
        print(f"Mean Macro F1-Score: {mean_score:.4f}")

    best_name = max(model_scores, key=model_scores.get)
    return best_name, best_pipelines[best_name], model_scores[best_name]

def run_default(
        X_train, y_train,
        X_val, y_val,
        models, model_dict,
        imbalance, le
    ):
    """Handles standard Train/Val split training and plotting."""
    model_scores, best_pipelines = {}, {}

    for model_name in models:
        print(f"\n{'='*50}\nTraining {model_name.upper()}...\n{'='*50}")
        start = time.time()
        
        estimator_to_use = build_pipeline(model_dict[model_name], imbalance)
        estimator_to_use.fit(X_train, y_train)
        
        y_pred = estimator_to_use.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True, digits=3)
        
        model_scores[model_name] = report['macro avg']['f1-score']
        best_pipelines[model_name] = estimator_to_use

        print(f"Time taken: {time.time() - start:.2f} seconds")
        print(f"\nClassification Report for {model_name.upper()}:")
        print(classification_report(y_val, y_pred, digits=3))

        # Plotting logic
        cm = confusion_matrix(y_val, y_pred)
        class_names = le.inverse_transform(np.arange(len(le.classes_))) if le else None
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f"Confusion Matrix for {model_name.upper()}")
        # plt.show()

    best_name = max(model_scores, key=model_scores.get)
    return best_name, best_pipelines[best_name], model_scores[best_name]


def train_and_evaluate_models(
        X_train_processed, y_train_enc,
        X_val_processed, y_val_enc,
        models, cv=False, grid_config_file=None, imbalance=False, 
        le=None, models_to_combine=None
    ):
    
    # 1. Get dictionary with dynamic ensembles
    model_dict = get_model_dict(models_to_combine)

    # 2. Handle 'all' keyword
    if models == ['all']:
        exclude_models = ['gb', 'svc', 'lsvc', 'qda', 'dummy']
        models = [m for m in model_dict.keys() if m not in exclude_models]
        
    # Optional: Automatically add the custom ensembles to the 'models' list to train them
    if models_to_combine:
        if 'custom_voting' not in models: models.append('custom_voting')
        if 'custom_stacking' not in models: models.append('custom_stacking')

    # 3. Route to the correct training loop
    if grid_config_file:
        best_name, best_model, best_score = run_grid_search(
            X_train_processed, y_train_enc,
            models, model_dict, grid_config_file, cv, imbalance
        )
    elif cv:
        best_name, best_model, best_score = run_cv(
            X_train_processed, y_train_enc,
            models, model_dict, cv, imbalance
        )
    else:
        best_name, best_model, best_score = run_default(
            X_train_processed, y_train_enc,
            X_val_processed, y_val_enc, models, model_dict, imbalance, le
        )

    # 4. Final output and Refit
    print(f"\n{'='*50}")
    print(f"Overall Best Model: {best_name.upper()} with Macro F1-Score: {best_score:.4f}")
    print('='*50)
    print("Refitting the best model on the combined train and validation sets...")
    
    best_model.fit(
        np.concatenate([X_train_processed, X_val_processed]),
        np.concatenate([y_train_enc, y_val_enc])
    )

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
    models_to_combine = args.models_to_combine
    imbalance = args.imbalance

    train, test, test_id = load_data(
        train_path, test_path, drop_columns
    )
    target_variable = args.target_variable or train.columns[-1]

    if args.eda:
        eda(train, target_variable)

    if models:
        (
            X_train_processed,
            X_val_processed,
            y_train_enc,
            y_val_enc,
            le,
            preprocessor
        ) = preprocess_data(train, target_variable, test_size)

### Using positional arguments up to the last 5?
        best_model = train_and_evaluate_models(
            X_train_processed=X_train_processed, 
            y_train_enc=y_train_enc,
            X_val_processed=X_val_processed, 
            y_val_enc=y_val_enc, 
            models=models,
            cv=cv, 
            grid_config_file=grid_config_file, 
            imbalance=imbalance, 
            le=le,
            models_to_combine=models_to_combine  # Pass it right in!
        )

        test_predictions = predict_test_data(test, preprocessor, best_model, le)
        

        # Save predictions to a file or return them
        output = pd.DataFrame({'id': test_id, 'Irrigation_Need': test_predictions})
        output.to_csv(output_name, index=False)
        print(f"Predictions saved to {output_name}")

    print('Done!')

if __name__ == '__main__':
    main()
