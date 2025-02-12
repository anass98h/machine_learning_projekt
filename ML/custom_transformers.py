from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CombineCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, correlatedColumns):
        self.correlatedColumns = correlatedColumns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_combined = pd.DataFrame(X.copy(), columns=X.columns)

        for col1_idx, col2_idx in self.correlatedColumns:
            col1_name = X_combined.columns[col1_idx]
            col2_name = X_combined.columns[col2_idx]
            
            new_col_name = f"combined_{col1_name}_{col2_name}"
            X_combined[new_col_name] = X_combined[col1_name] * X_combined[col2_name]
                
        return X_combined


class FeatureWeights(BaseEstimator, TransformerMixin):
    def __init__(self, feature_weights):
        self.feature_weights = feature_weights

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        xWeighted = X.copy()
        for colIdx, weight in self.feature_weights.items():
            if colIdx < len(xWeighted.columns):
                colName = xWeighted.columns[colIdx]
                xWeighted[colName] *= weight
                #print(f"Column {colName} weighted by {weight}")
            else:
                print(f"Column {colIdx} not found in the dataset")

        return xWeighted


class symmetricalColumns(BaseEstimator, TransformerMixin):
    def __init__(self, symmetricalColumns):
        self.symmetricalColumns = symmetricalColumns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_sym = pd.DataFrame(X.copy(), columns=X.columns)

        for col1_idx, col2_idx in self.symmetricalColumns:
            col1_name = X_sym.columns[col1_idx]
            col2_name = X_sym.columns[col2_idx]
            
            new_col_name = f"sym_{col1_name}_{col2_name}"
            X_sym[new_col_name] = X_sym[col1_name] + X_sym[col2_name]
                
        return X_sym