"""Data loading and preprocessing module."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import Config


class DataLoader:
    """Load and preprocess the WiFi fingerprinting dataset."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._raw_df: pd.DataFrame | None = None
        self._processed_df: pd.DataFrame | None = None
    
    def load_data(self, path: Path | None = None) -> pd.DataFrame:
        """Load the raw dataset from CSV."""
        data_path = path or self.config.data_path
        self._raw_df = pd.read_csv(data_path)
        return self._raw_df.copy()
    
    def get_raw_data(self) -> pd.DataFrame:
        """Get the raw unprocessed data."""
        if self._raw_df is None:
            self.load_data()
        return self._raw_df.copy()
    
    def preprocess(
        self,
        df: pd.DataFrame | None = None,
        scale_features: bool = True,
        encode_labels: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Preprocess the dataset for ML training."""
        if df is None:
            df = self.get_raw_data()
        
        X = df[self.config.feature_columns].values.astype(np.float32)
        y_raw = df[self.config.target_column].values
        classes = list(df[self.config.target_column].unique())
        
        if scale_features:
            X = self.scaler.fit_transform(X)
        
        if encode_labels:
            y = self.label_encoder.fit_transform(y_raw)
        else:
            y = y_raw
        
        self._processed_df = df
        return X, y, classes
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and test sets."""
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=stratify_param,
        )
        return X_train, X_test, y_train, y_test
    
    def split_train_val_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training, validation, and test sets."""
        stratify_param = y if stratify else None
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=stratify_param,
        )
        
        val_ratio = self.config.validation_size / (1 - self.config.test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.config.random_seed,
            stratify=stratify_temp,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_label_mapping(self) -> dict[int, str]:
        """Get mapping from encoded labels to original class names."""
        return dict(enumerate(self.label_encoder.classes_))
    
    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original class names."""
        return self.label_encoder.inverse_transform(y_encoded)
