import os
import pandas as pd
from utils.load_dataset import load_arff, load_csv


def preprocess_dataset(df, dataset_type):
    if dataset_type == 'arff':
        df['text'] = df['document']
        df['label'] = df['class'].map({'yes': 1, 'no': 0})
    elif dataset_type == 'csv':
        df['text'] = df['instagram_comments']
        df['label'] = df['offensive_language']

    return df[['text', 'label']]


def load_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.arff':
        df = load_arff(file_path)
        return preprocess_dataset(df, 'arff')
    elif file_extension == '.csv':
        df = load_csv(file_path)
        return preprocess_dataset(df, 'csv')
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def inspect_dataset(df, name):
    print(f"Dataset: {name}")
    print(f"Number of instances: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print("\nFeatures:")
    for column in df.columns:
        print(f"- {column}: {df[column].dtype}")

    print("\nFirst few rows:")
    print(df.head())

    print("\nSummary statistics:")
    print(df.describe(include='all'))

    print("\nClass distribution:")
    print(df['label'].value_counts(normalize=True))

    print("\n" + "=" * 50 + "\n")


def merge_datasets(dfs):
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df


def load_and_merge_datasets(file_paths, inspect=True):
    dfs = []
    for file_path in file_paths:
        df = load_file(file_path)
        if inspect:
            inspect_dataset(df, os.path.basename(file_path))
        dfs.append(df)

    final_df = merge_datasets(dfs)

    if inspect:
        print("Merged Dataset:")
        print(f"Number of instances: {len(final_df)}")
        print(f"Number of features: {len(final_df.columns)}")
        print("\nFirst few rows:")
        print(final_df.head())
        print("\nClass distribution:")
        print(final_df['label'].value_counts(normalize=True))

    return final_df


if __name__ == "__main__":
    offcombr2_path = '../dataset/OffComBR2.arff'
    offcombr3_path = '../dataset/OffComBR3.arff'
    hatebr_path = '../dataset/HateBR.csv'

    all_paths = [offcombr2_path, offcombr3_path, hatebr_path]
    load_and_merge_datasets(all_paths)
