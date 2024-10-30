import pandas as pd

# Read the original file
with open('../dataset/AI_generated.csv', 'r', encoding='utf-8') as file:
    texts = file.readlines()

# Clean the texts (remove newlines and strip whitespace)
texts = [text.strip() for text in texts if text.strip()]

# Manual approach
with open('../dataset/AI_processed_dataset.csv', 'w', encoding='utf-8') as f:
    f.write('text,label\n')  # header
    for text in texts:
        f.write(f'"{text}",1\n')
