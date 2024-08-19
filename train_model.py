import os
import argparse
import optuna
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

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

PARAMETER_TUNING = False
CROSS_VALIDATION = False


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


def cross_validate(args, train_val_df):
    ModelClass, process_dataset, train_model = get_model_and_functions(args.model)

    if CROSS_VALIDATION:
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
            print(f"Training fold {fold + 1}/{args.n_splits}")

            train_data = train_val_df.iloc[train_idx].reset_index(drop=True)
            val_data = train_val_df.iloc[val_idx].reset_index(drop=True)

            train_dataloader, val_dataloader = process_dataset(pd.concat([train_data, val_data]),
                                                               test_size=len(val_data) / (
                                                                           len(train_data) + len(val_data)),
                                                               batch_size=args.batch_size)

            model = ModelClass()

            # Train the model and get the best F1 score
            _, best_f1 = train_model(model, train_dataloader, val_dataloader,
                                     epochs=args.epochs, learning_rate=args.learning_rate,
                                     patience=3)

            fold_scores.append(best_f1)
            print(f"Fold {fold + 1} Best Validation F1 Score: {best_f1:.4f}")

        print(f"Validation F1 Scores: {fold_scores}")
        print(f"Average Validation F1 Score: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

        return np.mean(fold_scores)
    else:
        # If not doing cross-validation, just train once on the entire dataset
        train_dataloader, val_dataloader = process_dataset(train_val_df,
                                                           test_size=args.test_size,
                                                           batch_size=args.batch_size)
        model = ModelClass()
        _, best_f1 = train_model(model, train_dataloader, val_dataloader,
                                 epochs=args.epochs, learning_rate=args.learning_rate,
                                 patience=3)
        return best_f1


def objective(trial, train_val_df, model_name):
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 2, 15)

    class Args:
        def __init__(self):
            self.model = model_name
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.n_splits = 3
            self.test_size = 0.2

    args = Args()
    score = cross_validate(args, train_val_df)

    return score


def hyperparameter_tuning(train_val_df, model_name, n_trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train_val_df, model_name), n_trials=n_trials)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params


def main(args):
    print(f"Initializing {args.model.upper()} classifier...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Load train+validation data
    train_val_df = pd.read_csv(args.data_path)

    if PARAMETER_TUNING:
        # Perform hyperparameter tuning
        print("Starting hyperparameter tuning...")
        best_params = hyperparameter_tuning(train_val_df, args.model, n_trials=args.n_trials)

        # Update args with best hyperparameters
        args.learning_rate = best_params['learning_rate']
        args.batch_size = best_params['batch_size']
        args.epochs = best_params['epochs']
        args.test_size = 0

    if CROSS_VALIDATION:
        # Perform cross-validation with best hyperparameters
        print("Performing cross-validation with best hyperparameters...")
        mean_cv_score = cross_validate(args, train_val_df)

    # Train final model with best hyperparameters
    ModelClass, process_dataset, train_model = get_model_and_functions(args.model)

    print(f"Initializing {args.model.upper()} classifier for final training...")
    model = ModelClass()

    print("Processing entire dataset...")
    train_dataloader, val_dataloader = process_dataset(train_val_df, test_size=args.test_size,
                                                       batch_size=args.batch_size)

    print("Starting final model training...")
    final_model, _ = train_model(model, train_dataloader, val_dataloader,
                                 epochs=args.epochs, learning_rate=args.learning_rate)

    print("Saving trained model...")
    torch.save(final_model.state_dict(), args.output_path)

    print(f"Training complete. Model saved as '{args.output_path}'")
    if PARAMETER_TUNING:
        print(f"Best Hyperparameters: {best_params}")
    if CROSS_VALIDATION:
        print(f"Mean Cross-Validation F1 Score with best hyperparameters: {mean_cv_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERTimbau-based hate speech classifier with hyperparameter "
                                                 "tuning and cross-validation")
    parser.add_argument("--model", type=str, default="xlm-roberta",
                        choices=['bert', 'roberta', 'xlm-roberta', 'bertimbau'],
                        help="Type of model to use (bert, roberta, xlm-roberta, or bertimbau)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data_path", type=str, default="dataset/train_val_data.csv",
                        help="Path to the training and validation data CSV file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the trained model. If not provided, a default name will be used.")
    parser.add_argument("--n_splits", type=int, default=3, help="Number of splits for cross-validation")
    parser.add_argument("--n_trials", type=int,
                        default=50, help="Number of trials for hyperparameter tuning")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for training")

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"{args.model}_hatespeech_classifier.pth"

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"The specified data file does not exist: {args.data_path}")

    main(args)
