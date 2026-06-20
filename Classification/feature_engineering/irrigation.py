"""
Feature engineering for the irrigation classification project.
"""


class IrrigationFeatureEngineer:

    def transform(self, train, test):
        for data in (train, test):
            pass

        return train, test

    def encoding_config(self):
        return {
            'ordinal_cols': ['Soil_Type', 'Crop_Growth_Stage'],
            'ordinal_mappings': [
                ['Sandy', 'Loamy', 'Silt', 'Clay'],
                ['Sowing', 'Vegetative', 'Flowering', 'Harvest'],
            ],
            
            'binary_cols': ['Mulching_Used'],
            'binary_mappings': [['No', 'Yes']],
        }