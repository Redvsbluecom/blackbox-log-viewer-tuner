import numpy as np

def extract_features_from_inav_log(df):
    features = {}

    for axis, name in zip(range(3), ['roll', 'pitch', 'yaw']):
        rc_col = f'rcCommand[{axis}]'
        rate_col = f'axisRate[{axis}]'

        if rc_col not in df.columns or rate_col not in df.columns:
            features[f'{name}_overshoot'] = 0
            features[f'{name}_avg_error'] = 0
            features[f'{name}_oscillation_count'] = 0
            continue

        rc = df[rc_col]
        rate = df[rate_col]
        error = rc - rate

        features[f'{name}_overshoot'] = max((rate - rc).max(), 0)
        features[f'{name}_avg_error'] = error.abs().mean()
        features[f'{name}_oscillation_count'] = ((rate.diff().apply(np.sign)).diff().abs() > 1).sum()

    return features
