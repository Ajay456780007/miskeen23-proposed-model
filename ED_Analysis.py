from imblearn.over_sampling import SMOTE
import pandas as pd

# Assume X contains features, y contains the target labels
# Example: X = df.drop('target', axis=1), y = df['target']

def apply_smote(X, y, sampling_strategy='auto', random_state=42):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Convert back to DataFrame (optional, to retain column names)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)

    return X_resampled, y_resampled


