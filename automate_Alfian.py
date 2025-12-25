import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load raw dataset"""
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values based on domain knowledge"""
    df = df.copy()
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows"""
    return df.drop_duplicates()


def handle_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Clip outliers using IQR method"""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = numeric_cols.drop("is_canceled")

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)

    return df


def apply_binning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply binning to selected numerical features"""
    df = df.copy()

    df["lead_time_bin"] = pd.cut(
        df["lead_time"],
        bins=[-1, 30, 180, df["lead_time"].max()],
        labels=["short", "medium", "long"]
    )

    df["adr_bin"] = pd.qcut(
        df["adr"],
        q=3,
        labels=["low_price", "mid_price", "high_price"]
    )

    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove target leakage columns"""
    leakage_cols = ["reservation_status", "reservation_status_date"]
    return df.drop(columns=leakage_cols)


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features"""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    X_encoded = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True
    )
    return X_encoded


def scale_features(X: pd.DataFrame) -> np.ndarray:
    """Standardize features"""
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def split_data(
    X: np.ndarray,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Split dataset into train and test"""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def save_preprocessed_data(
    output_dir: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series
):
    """Save preprocessed dataset to .npy files"""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test.values)


def run_preprocessing_pipeline(
    input_path: str,
    output_dir: str
):
    """Main preprocessing pipeline"""
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers_iqr(df)
    df = apply_binning(df)
    df = drop_leakage_columns(df)

    X = df.drop("is_canceled", axis=1)
    y = df["is_canceled"]

    X_encoded = encode_categorical_features(X)
    X_scaled = scale_features(X_encoded)

    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    save_preprocessed_data(
        output_dir,
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    run_preprocessing_pipeline(
        input_path="hotel_bookings.csv",
        output_dir="preprocessing/namadataset_preprocessing"
    )