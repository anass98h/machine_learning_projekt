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
        """
        feature_weights: dictionary where keys are column indices (of the original DataFrame)
                         and values are the number of times that column should appear in total.
        For example, if a column has a weight of 4, it will appear once in its original position,
        and three extra duplicate columns will be appended.
        """
        self.feature_weights = feature_weights

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure we have a DataFrame with the original column order.
        X_in = pd.DataFrame(X.copy(), columns=X.columns)
        
        # Create a DataFrame to hold the duplicate columns.
        duplicates = pd.DataFrame(index=X_in.index)
        
        # Iterate over the original columns and create duplicates as needed.
        for i, col in enumerate(X_in.columns):
            if i in self.feature_weights:
                total_count = self.feature_weights[i]
                # Only add duplicates if total_count > 1 (since original column remains)
                for j in range(total_count - 1):
                    new_col_name = f"{col}_dup{j+1}"
                    duplicates[new_col_name] = X_in[col]
        
        # Append the duplicate columns to the end of the original DataFrame.
        X_out = pd.concat([X_in, duplicates], axis=1)
        return X_out


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
    
class SquareFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns, replace=False):
        """
        Parameters:
        - columns: list of column names to square.
        - replace: if True, overwrite the original column with its square.
                   If False (default), add a new column with a '_squared' suffix.
        """
        self.columns = columns
        self.replace = replace

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a DataFrame (this transformer expects a DataFrame)
        X_transformed = pd.DataFrame(X.copy(), columns=X.columns)
        for col in self.columns:
            if col in X_transformed.columns:
                squared = X_transformed[col] ** 2
                if self.replace:
                    X_transformed[col] = squared
                else:
                    X_transformed[f"{col}_squared"] = squared
        return X_transformed
