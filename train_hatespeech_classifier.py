import torch
import argparse

from model.bert_classifier import (BertClassifier,
                                   process_dataset as process_dataset_bert,
                                   train_model as train_model_bert)
from model.roberta_classifier import (RobertaClassifier,
                                      process_dataset as process_dataset_roberta,
                                      train_model as train_model_roberta)
from model.xlm_roberta_classifier import (XLMRobertaClassifier,
                                          process_dataset as process_dataset_xlm_roberta,
                                          train_model as train_model_xlm_roberta)
from utils.preprocess_dataset import load_and_merge_datasets


def get_model_and_functions(model_name):
    if model_name.lower() == 'bert':
        return BertClassifier, process_dataset_bert, train_model_bert
    elif model_name.lower() == 'roberta':
        return RobertaClassifier, process_dataset_roberta, train_model_roberta
    elif model_name.lower() == 'xlm-roberta':
        return XLMRobertaClassifier, process_dataset_xlm_roberta, train_model_xlm_roberta
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main(args):
    # Load and merge your datasets
    all_paths = [args.offcombr2_path, args.offcombr3_path, args.hatebr_path]

    print("Loading and merging datasets...")
    merged_df = load_and_merge_datasets(all_paths, inspect=False)

    # Get the appropriate model and process function
    ModelClass, process_dataset, train_model = get_model_and_functions(args.model)

    # Initialize the model
    print(f"Initializing {args.model.upper()} classifier...")
    model = ModelClass()

    # Process the dataset and get dataloaders
    print("Processing dataset...")
    train_dataloader, val_dataloader = process_dataset(merged_df, test_size=0.2, batch_size=args.batch_size)

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
    parser.add_argument("--model", type=str, default="xlm-roberta", choices=['bert', 'roberta', 'xlm-roberta'],
                        help="Type of model to use (bert, roberta, or xlm-roberta)")
    parser.add_argument("--offcombr2_path", type=str, default="dataset/OffComBR2.arff",
                        help="Path to OffComBR2 dataset")
    parser.add_argument("--offcombr3_path", type=str, default="dataset/OffComBR3.arff",
                        help="Path to OffComBR3 dataset")
    parser.add_argument("--hatebr_path", type=str, default="dataset/HateBR.csv", help="Path to HateBR dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the trained model. If not provided, a default name will be used.")

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output_path is None:
        args.output_path = f"{args.model}_hatespeech_classifier.pth"

    main(args)
