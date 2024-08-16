import os
import pandas as pd
from sklearn.model_selection import train_test_split

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


def merge_datasets(dfs):
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df


def load_and_merge_datasets(file_paths, test_size=0.2, random_state=42):
    dfs = []
    for file_path in file_paths:
        df = load_file(file_path)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    train_val_df, test_df = train_test_split(merged_df, test_size=test_size, random_state=random_state,
                                             stratify=merged_df['label'])

    return train_val_df, test_df


if __name__ == "__main__":
    offcombr2_path = '../dataset/OffComBR2.arff'
    offcombr3_path = '../dataset/OffComBR3.arff'
    hatebr_path = '../dataset/HateBR.csv'
    all_paths = [offcombr2_path, offcombr3_path, hatebr_path]

    train_val_df, test_df = load_and_merge_datasets(all_paths)

    print("Train+Validation Dataset:")
    print(f"Number of instances: {len(train_val_df)}")
    print("\nClass distribution:")
    print(train_val_df['label'].value_counts(normalize=True))

    print("\nTest Dataset:")
    print(f"Number of instances: {len(test_df)}")
    print("\nClass distribution:")
    print(test_df['label'].value_counts(normalize=True))

    # Save the datasets
    train_val_df.to_csv('../dataset/train_val_data.csv', index=False)
    test_df.to_csv('../dataset/test_data.csv', index=False)
