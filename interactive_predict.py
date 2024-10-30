import warnings

# Suppress the Flash Attention warning using a valid regex pattern
warnings.filterwarnings(
    "ignore",
    message=".*Torch was not compiled with flash attention.*",
    category=UserWarning
)

# Suppress the `clean_up_tokenization_spaces` FutureWarning
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning
)

import torch
from transformers import AutoTokenizer
from model.bertimbau_classifier import BERTimbauClassifier  # Ensure this is correctly defined
import re

# ----------------------------
# Preprocessing Functions
# ----------------------------

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

    # text = remove_mentions_hashtags(text)  # Step 1: Remove @mentions and #hashtags
    text = remove_urls(text)  # Step 2: Remove URLs and isolated 'http/https'
    # text = normalize_text(text, SUBSTITUTION_MAP)  # Step 3: Normalize text
    # text = remove_punctuation(text)  # Step 4: Remove punctuation and underscores
    text = to_lowercase(text)  # Step 5: Convert to lowercase
    text = remove_rt(text)  # Step 6: Remove standalone 'rt'
    # Optionally, trim extra whitespace
    # text = re.sub(r'\s+', ' ', text).strip()

    return text if text else ''  # Ensure non-empty


# ----------------------------
# Prediction Functions
# ----------------------------

def get_model(model_path, device):
    """
    Loads the trained BERTimbau model with weights_only=True.
    """
    model = BERTimbauClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def get_tokenizer():
    """
    Loads the BERTimbau tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    return tokenizer


def predict(text, model, tokenizer, device, max_length=128):
    """
    Predicts whether the input text is offensive (1) or not (0).
    """
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs
        prediction = torch.argmax(logits, dim=1).item()

    return prediction


# ----------------------------
# Interactive Prediction
# ----------------------------

def interactive_cli(model, tokenizer, device):
    """
    Runs an interactive command-line interface for predictions.
    """
    print("=== Hate Speech Detection Interactive CLI ===")
    print("Type 'exit' to quit the program.\n")

    while True:
        user_input = input("Enter a sentence to analyze: ")

        if user_input.strip().lower() == 'exit':
            print("Exiting the program.")
            break

        # Ask if the user wants to preprocess the input
        preprocess_choice = input("Do you want to preprocess the input? (yes/no): ").strip().lower()

        if preprocess_choice in ['yes', 'y']:
            processed_text = preprocess_text(user_input)
            if not processed_text:
                print("The input text is empty after preprocessing. Please try again with different text.\n")
                continue
            print(f"Preprocessed Text: {processed_text}")
            text_to_predict = processed_text
        elif preprocess_choice in ['no', 'n']:
            text_to_predict = user_input
        else:
            print("Invalid choice for preprocessing. Please enter 'yes' or 'no'.\n")
            continue

        # Make the prediction
        prediction = predict(text_to_predict, model, tokenizer, device)
        label = prediction  # 0: Not Offensive, 1: Offensive
        label_str = 'Offensive' if label == 1 else 'Not Offensive'
        print(f"Prediction: {label} ({label_str})\n")


# ----------------------------
# Main Function
# ----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Hate Speech Prediction Tool")
    parser.add_argument(
        "--model_path",
        type=str,
        default="bertimbau_hatespeech_classifier.pth",
        help="Path to the trained BERTimbau model weights"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    print("Loading model...")
    model = get_model(args.model_path, device)
    tokenizer = get_tokenizer()
    print("Model and tokenizer loaded successfully.\n")

    # Start interactive CLI
    interactive_cli(model, tokenizer, device)


if __name__ == "__main__":
    main()
