import os
import pandas as pd
from src.config import DATA_PATH, CSV_FILES


def load_feature_names():
    feature_path = os.path.join(DATA_PATH, "NUSW-NB15_features.csv")
    try:
        features_df = pd.read_csv(feature_path, encoding="utf-8")
    except UnicodeDecodeError:
        features_df = pd.read_csv(feature_path, encoding="cp1252")
    feature_names = features_df["Name"].tolist()

    # Avoid duplicating the label column; the raw CSV already includes it
    lower_names = [name.lower() for name in feature_names]
    if "label" not in lower_names:
        feature_names.append("label")
    return feature_names


def load_and_combine_data():
    feature_names = load_feature_names()
    dataframes = []

    for file in CSV_FILES:
        file_path = os.path.join(DATA_PATH, file)

        df = pd.read_csv(
            file_path,
            header=None,              # VERY IMPORTANT
            names=feature_names,      # Use correct names
            low_memory=False
        )

        print(f"{file} shape: {df.shape}")
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df
