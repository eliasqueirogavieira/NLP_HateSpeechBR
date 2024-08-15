import pandas as pd


def load_arff(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    start_data = False
    for line in lines:
        if line.strip() == '@data':
            start_data = True
            continue
        if start_data and line.strip():
            # Split the line into class and document
            class_label, document = line.strip().split(',', 1)
            # Remove quotes from class_label and document
            class_label = class_label.strip("'")
            document = document.strip("'")
            data.append([class_label, document])

    df = pd.DataFrame(data, columns=['class', 'document'])
    return df


def load_csv(file_path):
    return pd.read_csv(file_path)
