# Portuguese Hate Speech Detection

This repository contains a machine learning project for detecting hate speech in Portuguese text using various transformer-based models.

## Project Structure

- `dataset/`: Contains the dataset files
- `model/`: Contains the model implementations
- `utils/`: Contains utility functions

## Setup

1. Clone this repository:
```
git clone https://github.com/eliasqueirogavieira/NLP_HateSpeechBR.git
cd NLP_HateSpeechBR
```

2. Install the required packages:
```
pip install -r requirements.txt
```

## Training the Model

To train the model, use the `train_model.py` script. You can customize various parameters:
```
python train_model.py [OPTIONS]
```

Options:
- `--model`: Type of model to use (bert, roberta, xlm-roberta, or bertimbau). Default is "bertimbau".
- `--epochs`: Number of training epochs. Default is 5.
- `--learning_rate`: Learning rate. Default is 2e-5.
- `--batch_size`: Batch size. Default is 32.
- `--data_path`: Path to the training and validation data CSV file. Default is "dataset/train_val_data.csv".
- `--output_path`: Path to save the trained model. If not provided, a default name will be used.

Example:
```
python train_model.py --model bertimbau --epochs 5 --learning_rate 2e-5 --batch_size 32
```

## Evaluating the Model

To evaluate a trained model, use the `evaluate_model.py` script:
```
python evaluate_model.py [OPTIONS]
```

Options:
- `--model`: Type of model to evaluate (bert, roberta, xlm-roberta, or bertimbau). Default is "bertimbau".
- `--model_path`: Path to the trained model weights. Default is "bertimbau_hatespeech_classifier.pth".
- `--data_path`: Path to the testing data CSV file. Default is "dataset/test_data.csv".
- `--batch_size`: Batch size for evaluation. Default is 32.

Example:
```
python evaluate_model.py --model bertimbau --model_path bertimbau_hatespeech_classifier.pth
```


## Models

The project supports the following models:
- BERT
- RoBERTa
- XLM-RoBERTa
- BERTimbau (Portuguese BERT)

## Dataset

The dataset should be split into two CSV files:
- `dataset/train_val_data.csv`: For training and validation
- `dataset/test_data.csv`: For final evaluation

Each CSV should contain 'text' and 'label' columns.

## Results

After training, the model will be saved in the specified output path or with a default name based on the model type.

Evaluation results will display accuracy, F1 score, and a detailed classification report.

## Notes

- Ensure you have sufficient GPU resources for training larger models.
- Adjust batch size based on your GPU memory capacity.
- For best results with Portuguese text, the BERTimbau model is recommended.
