import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from utils.load_dataset import load_arff, load_csv
import unicodedata


def remove_special_characters(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Normalize the text to decomposed form
    text = unicodedata.normalize('NFD', text)

    # Remove any character that is not a letter, number, or allowed punctuation
    # The \u0300-\u036f range covers combining diacritical marks
    pattern = r'[^\w\s.,;çÇ\-$%!?\u0300-\u036f]'
    text = re.sub(pattern, '', text, flags=re.UNICODE)

    # Normalize back to composed form
    text = unicodedata.normalize('NFC', text)

    return text.strip()  # Remove leading/trailing whitespace


def preprocess_dataset(df, dataset_type):
    if dataset_type == 'arff':
        df['text'] = df['document']
        df['label'] = df['class'].map({'yes': 1, 'no': 0})
    elif dataset_type == 'csv':
        df['text'] = df['text']
        df['label'] = df['label']

    # Remove special characters
    #df['text'] = df['text'].apply(remove_special_characters)

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


def load_and_merge_datasets(file_paths, test_size=0.2, random_state=42):
    dfs = []
    for file_path in file_paths:
        df = load_file(file_path)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    # Check for duplicates
    duplicate_count = merged_df.duplicated().sum()
    print(f"Number of duplicates found: {duplicate_count}")

    # Remove duplicates
    merged_df.drop_duplicates(inplace=True)
    print(f"Number of instances after removing duplicates: {len(merged_df)}")

    train_val_df, test_df = train_test_split(merged_df, test_size=test_size, random_state=random_state,
                                             stratify=merged_df['label'])

    return train_val_df, test_df


if __name__ == "__main__":
    offcombr2_path = '../dataset/OffComBR2.arff'
    offcombr3_path = '../dataset/OffComBR3.arff'
    hatebr_path = '../dataset/HateBR.csv'
    kaggleds_path = '../dataset/kaggle_dataset.csv'
    all_paths = [offcombr2_path, offcombr3_path, hatebr_path, kaggleds_path]

    train_val_df, test_df = load_and_merge_datasets(all_paths)

    print("\nTrain+Validation Dataset:")
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

    # Print a few examples of preprocessed text
    print("\nExamples of preprocessed text:")
    for i in range(5):
        print(f"Example {i+1}: {train_val_df['text'].iloc[i]}")
