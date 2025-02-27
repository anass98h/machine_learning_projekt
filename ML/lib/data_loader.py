import pandas as pd
import os
from typing import Tuple, Optional
import numpy as np

# Get the directory of this script for building relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default configuration 
DEFAULT_CONFIG = {
    "data_dir_relative": "../data",  # Relative to script directory
    "data_filename": "AimoScore_WeakLink_big_scores.xls",
    "target_column": 0,  # First column is target
    "feature_columns_start": 1,
    "feature_columns_end": -1,  # Use -1 to indicate all columns until the end
    "test_size": 0.2,
    "random_state": 42
}

def load_data(config: Optional[dict] = None) -> pd.DataFrame:
    """
    Load data from the specified file path using cross-platform compatible paths.
    
    Args:
        config: Dictionary with configuration parameters. If None, use default config.
    
    Returns:
        DataFrame containing the loaded data
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Make a copy of the default config and update with provided values
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    # If an absolute path is provided, use it directly
    if "data_path" in config and os.path.isabs(config["data_path"]):
        file_path = os.path.join(config["data_path"], config["data_filename"])
    else:
        # Otherwise use a path relative to the script directory
        data_dir = os.path.join(SCRIPT_DIR, config.get("data_dir_relative", DEFAULT_CONFIG["data_dir_relative"]))
        file_path = os.path.join(data_dir, config["data_filename"])
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    return pd.read_excel(file_path)

def get_features_and_target(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target from dataframe.
    
    Args:
        df: DataFrame containing the data
        config: Dictionary with configuration parameters. If None, use default config.
    
    Returns:
        Tuple of (features_df, target_series)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Make a copy of the default config and update with provided values
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    target = df.iloc[:, config["target_column"]]
    
    # Handle the case where feature_columns_end is -1 (all columns)
    if config["feature_columns_end"] == -1:
        features = df.iloc[:, config["feature_columns_start"]:]
    else:
        features = df.iloc[:, config["feature_columns_start"]:config["feature_columns_end"]]
    
    return features, target

def load_and_split_data(config: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data and split into train/test sets based on configuration.
    
    Args:
        config: Dictionary with configuration parameters. If None, use default config.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    df = load_data(config)
    X, y = get_features_and_target(df, config)
    
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Make a copy of the default config and update with provided values
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    return train_test_split(
        X, y, 
        test_size=config["test_size"], 
        random_state=config["random_state"]
    )