#!/usr/bin/env python

#TODO:
# 5. Plot the results for each model and compare them visually.
# 6. Add more evaluation metrics like ROC-AUC, Precision-Recall curves, etc.
# 7. Add correlation matrix to EDA
# 8. Add feature importance plots for tree-based models.
#
# DONE (refactor): create_new_features() is no longer hardcoded in this
# file. Project-specific feature engineering now lives in
# feature_engineering/<name>.py and is selected at runtime via
# --feature-module <name>. See feature_engineering/base.py for the
# interface and feature_engineering/stellar.py for an example
# implementation. Run e.g. --feature-module stellar for this dataset.

# Remaining hardcoded parameters (intentionally left as script-level
# config, not yet promoted to per-project plugin classes):
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
)
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
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
    AdaBoostClassifier, VotingClassifier, StackingClassifier
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
from sklearn.decomposition import PCA

from feature_engineering import load_feature_engineer

def command_line_args():
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description="""
            Generic Classification Pipeline:
            This script loads the training and test datasets,
            performs exploratory data analysis (EDA) if specified,
            applies optional project-specific feature engineering,
            and trains/evaluates one or more classification models.
            Usage:
            ./pipeline.py --train-dataset ../data/stellar/train.csv
            --test-dataset ../data/stellar/test.csv [--drop-columns id]
            [--feature-module stellar] [--eda]
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
        '--feature-module',
        type=str,
        default=None,
        help='Name of a project-specific feature engineering module '
             'in feature_engineering/ (e.g. "stellar"). Omit to run '
             'on raw columns with no custom feature engineering.'
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
        '--pca',
        action='store_true',
        help='Visualize PCA 2D projection of features'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.1,
        help='Proportion of the dataset to include in the test split'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=[
            'dt', 'rf', 'lr', 'ridge', 'sgd', 'pa', 'perceptron',
                'et', 'gb', 'hgb', 'xgb', 'lgbm', 'ada', 'svc', 'lsvc',
                'knn', 'gnb', 'qda', 'lda', 'mlp', 'dummy', 'cat',
                'rf_ovr', 'ann', 'all'
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
        '--combine-models-method',
        choices=['voting', 'stacking'],
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/result.csv',
        help='Path to save the predictions'
    )
    return parser.parse_args()

class ANNClassifier(BaseEstimator, ClassifierMixin):
    """A minimal sklearn-compatible PyTorch classifier wrapper.

    This trains a simple feed-forward network with one hidden layer and
    exposes `fit`, `predict` and `predict_proba` so it can be used alongside
    other sklearn estimators in the script.
    """
    def __init__(
            self,
            input_dim=None,
            hidden_dim=64,
            epochs=200,
            batch_size=32,
            lr=1e-3,
            verbose=False,
            plot_training_history=True,
            device=None
            ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.plot_training_history = plot_training_history
        self._model = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_history_ = []
        self.accuracy_history_ = []

    def _build_model(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def _plot_training_history(self):
        if not self.loss_history_ or not self.accuracy_history_:
            return

        epochs = np.arange(1, len(self.loss_history_) + 1)
        _, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, self.loss_history_, marker='o')
        axes[0].set_title('ANN Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.accuracy_history_, marker='o', color='tab:green')
        axes[1].set_title('ANN Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0.8, 1)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

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
        self.loss_history_ = []
        self.accuracy_history_ = []

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                out = self._model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                epoch_correct += (out.argmax(dim=1) == yb).sum().item()

            epoch_loss /= n_samples
            epoch_accuracy = epoch_correct / n_samples
            self.loss_history_.append(epoch_loss)
            self.accuracy_history_.append(epoch_accuracy)

            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{self.epochs}, "
                    f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
                )

        if self.plot_training_history:
            self._plot_training_history()

        return self

    def predict_proba(self, X):
        if torch is None:
            raise RuntimeError('PyTorch is not available. Install torch to use ANNClassifier')
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

def load_data(train_path, test_path, drop_columns=[]):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if 'id' in test.columns:
        test_id = test['id']
    else:
        print("No 'id' column found in test data; using row index instead.")
        test_id = test.index.to_series(name='id')

    for col in drop_columns:
        try:
            print(f'Dropping column: {col}')
            for data in [train, test]:
                data.drop(columns=[col], inplace=True)
        except:
            print(f'Could not delete column: {col} in train or test')

    return train, test, test_id

def apply_feature_engineering(train, test, feature_module):
    """Applies the project-specific feature engineering module, if any.

    Kept separate from load_data so callers can run EDA on raw columns
    before any derived features are added. Returns the feature_engineer
    instance too (or None), since preprocess_data may need it for
    project-specific encoding config (ordinal/binary column mappings).
    """

    if not feature_module:
        return train, test, None

    feature_engineer = load_feature_engineer(feature_module)
    print(f"Applying feature engineering module: {feature_module}")
    train, test = feature_engineer.transform(train, test)
    return train, test, feature_engineer


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
    n_cols_plot = 2
    n_rows_plot = int(np.ceil(len(num_cols) / n_cols_plot))
    print('\nBox Plots for Numerical Features:')
    _, axes1 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(14, 2.5*n_rows_plot))
    axes1 = np.atleast_1d(axes1).flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(train_data[col], ax=axes1[i], orient='h')
        axes1[i].set_title(f"Distribution of {col}")
    for j in range(len(num_cols), len(axes1)):
        axes1[j].axis('off')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout()

    plt.show()

    print('EDA Completed.')

def pca_2d_visualization(X_train_processed, y_train_enc, le=None, title="PCA 2D Visualization"):
    """
    Project features to 2D using PCA and visualize class distribution.
    
    Parameters:
    -----------
    X_train_processed : array-like
        Preprocessed feature matrix
    y_train_enc : array-like
        Encoded target labels
    le : LabelEncoder, optional
        Label encoder to convert numeric labels back to class names
    title : str
        Title for the plot
    """
    print('\nPerforming PCA visualization...')
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_processed)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get unique classes and create a color map
    unique_classes = np.unique(y_train_enc)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
    
    # Plot each class with a different color
    for idx, class_label in enumerate(unique_classes):
        mask = y_train_enc == class_label
        class_name = le.inverse_transform([class_label])[0] if le else f"Class {class_label}"
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[colors[idx]], label=class_name, alpha=0.1, s=50, edgecolors='k', linewidth=0.5
        )
    
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print('PCA visualization completed.')
    return pca, X_pca

def preprocess_data(train, target_variable, test_size, feature_engineer=None):
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

    config = feature_engineer.encoding_config() if hasattr(feature_engineer, 'encoding_config') else {}
    ordinal_cols = config.get('ordinal_cols', [])
    ordinal_mappings = config.get('ordinal_mappings', [])
    binary_cols = config.get('binary_cols', [])
    binary_mappings = config.get('binary_mappings', [])

    # One hot encoding for remaining categorical columns
    onehot_cols = [
        c for c in cat_cols if c not in ordinal_cols and c not in binary_cols
        ]

    transformers = [
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), onehot_cols),
        ('num', StandardScaler(), num_cols)
        ]
    
    if ordinal_cols:
        transformers.append(
            ('ordinal', OrdinalEncoder(categories=ordinal_mappings), ordinal_cols)
            )
    if binary_cols:
        transformers.append(
            ('binary', OrdinalEncoder(categories=binary_mappings), binary_cols)
            )

    # Create preprocessor with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        # [
        #     # ('ordinal', OrdinalEncoder(categories=ordinal_mappings), ordinal_cols),
        #     # ('binary', OrdinalEncoder(categories=[Mulching_Used_mapping]), binary_cols),
        #     ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), onehot_cols),
        #     ('num', StandardScaler(), num_cols)
        # ],
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


def get_model_dict():
    """Stores base models."""
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
        'ann': ANNClassifier(),
        # One vs Rest
        'rf_ovr': OneVsRestClassifier(
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        ),
    }

    return model_dict


def expand_model_list(models, model_dict):
    """Expand model aliases and validate model names."""
    if not models:
        raise ValueError('Use --models to choose which models should be trained.')

    if 'all' in models:
        exclude_models = ['gb', 'svc', 'lsvc', 'qda', 'dummy']
        return [
            name for name in model_dict
            if name not in exclude_models
        ]

    invalid_models = [model_name for model_name in models if model_name not in model_dict]
    if invalid_models:
        raise ValueError(f"Unknown model(s): {', '.join(invalid_models)}")

    return list(dict.fromkeys(models))


def add_combined_model(model_dict, models, combine_models_method):
    """Add a combined model built from the models passed to --models."""
    if not combine_models_method:
        return models

    if len(models) < 2:
        raise ValueError('--combine-models-method requires at least two models in --models.')

    estimators = [
        (model_name, clone(model_dict[model_name]))
        for model_name in models
    ]

    if combine_models_method == 'voting':
        models_without_proba = [
            model_name for model_name, estimator in estimators
            if not callable(getattr(estimator, 'predict_proba', None))
        ]
        voting_method = 'soft'
        print("Trying soft voting for the combined model...")

        if models_without_proba:
            voting_method = 'hard'
            print(
                'Soft voting is not possible because these models do not '
                f"support predict_proba: {', '.join(models_without_proba)}"
            )
            print("Falling back to hard voting.")

        model_dict['combined_voting'] = VotingClassifier(
            estimators=estimators,
            voting=voting_method,
            n_jobs=-1
        )
        return ['combined_voting']

    if combine_models_method == 'stacking':
        model_dict['combined_stacking'] = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight='balanced'),
            n_jobs=-1
        )
        return ['combined_stacking']

    raise ValueError(f"Unsupported combine models method: {combine_models_method}")


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
            cv=cv if cv else 5, scoring='f1_macro', n_jobs=1, verbose=2
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
        le=None, combine_models_method=None
    ):
    
    model_dict = get_model_dict()
    models = expand_model_list(models, model_dict)
    models = add_combined_model(model_dict, models, combine_models_method)

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
    feature_module = args.feature_module
    models = args.models
    output_name = args.output
    cv=args.cv
    grid_config_file=args.grid_config_file
    combine_models_method = args.combine_models_method
    imbalance = args.imbalance

    train, test, test_id = load_data(
        train_path, test_path, drop_columns
    )
    target_variable = args.target_variable or train.columns[-1]

    if args.eda:
        eda(train, target_variable)

    train, test, feature_engineer = apply_feature_engineering(train, test, feature_module)

    if models or args.pca:
        (
            X_train_processed,
            X_val_processed,
            y_train_enc,
            y_val_enc,
            le,
            preprocessor
        ) = preprocess_data(train, target_variable, test_size, feature_engineer)

        if args.pca:
            pca_2d_visualization(X_train_processed, y_train_enc, le=le, title="PCA 2D Visualization - Class Distribution")
        else:
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
                combine_models_method=combine_models_method
            )

            test_predictions = predict_test_data(test, preprocessor, best_model, le)
            

            # Save predictions to a file or return them
            output = pd.DataFrame({'id': test_id, target_variable: test_predictions})
            output.to_csv(output_name, index=False)
            print(f"Predictions saved to {output_name}")

    print('Done!')

if __name__ == '__main__':
    main()
