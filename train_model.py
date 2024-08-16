import os
import argparse

import pandas as pd
import torch

from model.bert_classifier import (BertClassifier,
                                   process_dataset as process_dataset_bert,
                                   train_model as train_model_bert)
from model.roberta_classifier import (RobertaClassifier,
                                      process_dataset as process_dataset_roberta,
                                      train_model as train_model_roberta)
from model.xlm_roberta_classifier import (XLMRobertaClassifier,
                                          process_dataset as process_dataset_xlm_roberta,
                                          train_model as train_model_xlm_roberta)
from model.bertimbau_classifier import (BERTimbauClassifier,
                                        process_dataset as process_dataset_bertimbau,
                                        train_model as train_model_bertimbau)


def get_model_and_functions(model_name):
    if model_name.lower() == 'bert':
        return BertClassifier, process_dataset_bert, train_model_bert
    elif model_name.lower() == 'roberta':
        return RobertaClassifier, process_dataset_roberta, train_model_roberta
    elif model_name.lower() == 'xlm-roberta':
        return XLMRobertaClassifier, process_dataset_xlm_roberta, train_model_xlm_roberta
    elif model_name.lower() == 'bertimbau':
        return BERTimbauClassifier, process_dataset_bertimbau, train_model_bertimbau
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main(args):
    # Load train+validation data
    train_val_df = pd.read_csv(args.data_path)

    # Get the appropriate model and process function
    ModelClass, process_dataset, train_model = get_model_and_functions(args.model)

    # Initialize the model
    print(f"Initializing {args.model.upper()} classifier...")
    model = ModelClass()

    # Process the dataset and get dataloaders
    print("Processing dataset...")
    train_dataloader, val_dataloader = process_dataset(train_val_df, test_size=0.2, batch_size=args.batch_size)

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Train the model
    print("Starting model training...")
    trained_model = train_model(model, train_dataloader, val_dataloader,
                                epochs=args.epochs, learning_rate=args.learning_rate)

    # Save the trained model
    print("Saving trained model...")
    torch.save(trained_model.state_dict(), args.output_path)

    print(f"Training complete. Model saved as '{args.output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer-based hate speech classifier")
    parser.add_argument("--model", type=str, default="bertimbau",
                        choices=['bert', 'roberta', 'xlm-roberta', 'bertimbau'],
                        help="Type of model to use (bert, roberta, xlm-roberta, or bertimbau)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data_path", type=str, default="dataset/train_val_data.csv",
                        help="Path to the training and validation data CSV file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the trained model. If not provided, a default name will be used.")

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"{args.model}_hatespeech_classifier.pth"

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"The specified data file does not exist: {args.data_path}")

    main(args)
