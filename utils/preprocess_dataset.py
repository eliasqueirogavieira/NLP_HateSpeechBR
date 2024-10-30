import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.load_dataset import load_arff, load_csv


# 1. Define the substitution mapping
SUBSTITUTION_MAP = {
    '4': 'a',
    '0': 'o',
    '3': 'e',
    '1': 'i',
    '!': 'i',
    '@': 'a',
    '#': 'h',
    '5': 's',
    '$': 's',
    '7': 't',
    '+': 't',
    '9': 'g',
    '8': 'b',  # Optional: depending on obfuscation patterns
    # Add more mappings as needed
}


# 2. Preprocessing Functions

def remove_mentions_hashtags(text):
    """
    Removes @mentions and #hashtags from the text.
    """
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove #hashtags
    text = re.sub(r'#\w+', '', text)
    return text


def remove_urls(text):
    """
    Removes URLs starting with http, https, or www from the text.
    Also removes isolated 'http' or 'https' if present.
    """
    # Remove URLs starting with http://, https://, or www.
    text = re.sub(r'(https?:\/\/\S+|www\.\S+)', '', text)
    # Remove isolated 'http' or 'https'
    text = re.sub(r'\bhttps?\b', '', text)
    return text


def remove_rt(text):
    """
    Removes the standalone word 'RT' from the text.
    """
    # Remove 'rt' when it's a standalone word
    text = re.sub(r'\brt\b', '', text)
    return text


def normalize_text(text, substitution_map):
    """
    Replaces numbers and symbols with corresponding letters based on the substitution map.
    """
    # Replace each character based on the substitution map
    return ''.join([substitution_map.get(char, char) for char in text])


def remove_punctuation(text):
    """
    Removes punctuation from the text except for the apostrophe.
    Also removes underscores.
    """
    # Remove all punctuation except apostrophes and remove underscores
    text = re.sub(r"[^\w\s']|_", '', text)
    return text


def to_lowercase(text):
    """
    Converts text to lowercase.
    """
    return text.lower()


def preprocess_text(text):
    """
    Applies all preprocessing steps to the input text.
    """
    if not isinstance(text, str) or text.lower() == 'nan':
        return ''  # Return empty string for invalid entries

    #text = remove_mentions_hashtags(text)  # Step 1: Remove @mentions and #hashtags
    text = remove_urls(text)  # Step 2: Remove URLs and isolated 'http/https'
    #text = normalize_text(text, SUBSTITUTION_MAP)  # Step 3: Normalize text
    #text = remove_punctuation(text)  # Step 4: Remove punctuation and underscores
    text = to_lowercase(text)  # Step 5: Convert to lowercase
    text = remove_rt(text)  # Step 6: Remove standalone 'rt'
    # Optionally, trim extra whitespace
    #text = re.sub(r'\s+', ' ', text).strip()

    return text if text else ''  # Ensure non-empty


def preprocess_dataset(df, dataset_type):
    """
    Preprocesses the dataset by selecting relevant columns and applying text preprocessing.
    """
    if dataset_type == 'arff':
        df['text'] = df['document']
        df['label'] = df['class'].map({'yes': 1, 'no': 0})
    elif dataset_type == 'csv':
        df['text'] = df['text']
        df['label'] = df['label']
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # **Step 1: Drop Rows with Missing 'text'**
    initial_count = len(df)
    df = df.dropna(subset=['text'])
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} entries due to missing 'text'.")

    # **Optional: Reset Index After Dropping Rows**
    df = df.reset_index(drop=True)

    # **Step 2: Apply Preprocessing to the 'text' Column**
    df['text'] = df['text'].astype(str).apply(preprocess_text)

    # **Step 3: Remove Entries with Empty 'text' After Preprocessing**
    invalid_entries = df[df['text'].str.strip() == '']
    if not invalid_entries.empty:
        print(f"Found {len(invalid_entries)} entries with empty 'text' after preprocessing. Dropping them.")
        df = df[df['text'].str.strip() != '']

    # **Optional: Reset Index Again**
    df = df.reset_index(drop=True)

    return df[['text', 'label']]


def load_file(file_path):
    """
    Loads a file and preprocesses it based on its extension.
    """
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
    """
    Loads multiple datasets, merges them, removes duplicates, and splits into train+validation and test sets.
    """
    dfs = []
    for file_path in file_paths:
        df = load_file(file_path)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    # **Check for Duplicates**
    duplicate_count = merged_df.duplicated().sum()
    print(f"Number of duplicates found: {duplicate_count}")

    # **Remove Duplicates**
    merged_df.drop_duplicates(inplace=True)
    print(f"Number of instances after removing duplicates: {len(merged_df)}")

    # **Split the Data**
    train_val_df, test_df = train_test_split(
        merged_df,
        test_size=test_size,
        random_state=random_state,
        stratify=merged_df['label']
    )

    return train_val_df, test_df


if __name__ == "__main__":
    # Define dataset paths
    offcombr2_path = '../dataset/OffComBR2.arff'
    offcombr3_path = '../dataset/OffComBR3.arff'
    hatebr_path = '../dataset/HateBR.csv'
    kaggleds_path = '../dataset/kaggle_dataset.csv'
    llmds_path = '../dataset/AI_processed_dataset.csv'
    all_paths = [offcombr2_path, offcombr3_path, hatebr_path, kaggleds_path, llmds_path]

    # Load and merge datasets
    train_val_df, test_df = load_and_merge_datasets(all_paths)

    # Display dataset information
    print("\nTrain+Validation Dataset:")
    print(f"Number of instances: {len(train_val_df)}")
    print("\nClass distribution:")
    print(train_val_df['label'].value_counts(normalize=True))

    print("\nTest Dataset:")
    print(f"Number of instances: {len(test_df)}")
    print("\nClass distribution:")
    print(test_df['label'].value_counts(normalize=True))

    # Save the preprocessed datasets
    train_val_df.to_csv('../dataset/train_val_data.csv', index=False)
    test_df.to_csv('../dataset/test_data.csv', index=False)

    # Print a few examples of preprocessed text
    print("\nExamples of preprocessed text:")
    for i in range(min(5, len(train_val_df))):
        print(f"Example {i + 1}: {train_val_df['text'].iloc[i]}")
