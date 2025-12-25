import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(csv_path):
    """
    Melakukan preprocessing data hotel bookings
    dan mengembalikan data train dan test siap latih
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Drop duplicate
    df = df.drop_duplicates()

    # Handle missing values
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)

    # Split feature dan target
    X = df.drop(columns=["is_canceled"])
    y = df["is_canceled"]

    # Encoding kategorikal
    X = pd.get_dummies(X, drop_first=True)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Gabungkan kembali
    train_df = X_train.copy()
    train_df["is_canceled"] = y_train.values

    test_df = X_test.copy()
    test_df["is_canceled"] = y_test.values

    return train_df, test_df