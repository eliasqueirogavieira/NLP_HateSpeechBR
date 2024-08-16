import torch
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from model.bert_classifier import BertClassifier, prepare_data
from model.roberta_classifier import RobertaClassifier
from model.xlm_roberta_classifier import XLMRobertaClassifier
from model.bertimbau_classifier import BERTimbauClassifier
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import argparse


def get_model_and_tokenizer(model_name):
    if model_name == 'bert':
        model = BertClassifier()
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'roberta':
        model = RobertaClassifier()
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    elif model_name == 'xlm-roberta':
        model = XLMRobertaClassifier()
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    elif model_name == 'bertimbau':
        model = BERTimbauClassifier()
        tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, tokenizer


def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    report = classification_report(all_labels, all_predictions)

    return accuracy, f1, report


def main(args):
    # Load the test data
    test_df = pd.read_csv(args.data_path)

    # Get the model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model)

    # Load the trained model weights
    model.load_state_dict(torch.load(args.model_path))

    # Prepare the test data
    test_dataset = prepare_data(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Evaluate the model
    accuracy, f1, report = evaluate_model(model, test_dataloader, device)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained hate speech classifier")
    parser.add_argument("--model", type=str, default="bertimbau",
                        choices=['bert', 'roberta', 'xlm-roberta', 'bertimbau'],
                        help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, default="bertimbau_hatespeech_classifier.pth",
                        help="Path to the trained model weights")
    parser.add_argument("--data_path", type=str, default="dataset/test_data.csv",
                        help="Path to the testing data CSV file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()
    main(args)
