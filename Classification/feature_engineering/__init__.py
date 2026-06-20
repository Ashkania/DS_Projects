import importlib


def load_feature_engineer(name: str):
    module = importlib.import_module(f'feature_engineering.{name}')
    class_name = ''.join(w.capitalize() for w in name.split('_')) + 'FeatureEngineer'
    return getattr(module, class_name)()