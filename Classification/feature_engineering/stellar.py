"""Feature engineering for the stellar classification project.

Derives photometric color indices and slope features from the SDSS-style
ugriz magnitude bands.
"""


class StellarFeatureEngineer:

    def transform(self, train, test):
        for data in (train, test):
            data['u_g'] = data['u'] - data['g']
            data['g_r'] = data['g'] - data['r']
            data['r_i'] = data['r'] - data['i']
            data['i_z'] = data['i'] - data['z']

            data['u_r'] = data['u'] - data['r']
            data['u_i'] = data['u'] - data['i']
            data['u_z'] = data['u'] - data['z']
            data['g_i'] = data['g'] - data['i']
            data['g_z'] = data['g'] - data['z']
            data['r_z'] = data['r'] - data['z']

            data['slope_ug'] = data['u_g'] - data['g_r']
            data['slope_gr'] = data['g_r'] - data['r_i']
            data['slope_ri'] = data['r_i'] - data['i_z']

        return train, test