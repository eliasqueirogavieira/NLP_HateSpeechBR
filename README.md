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
- Python 3.11 

#### Install the requirements:

```
pip install -r requirements.txt
```
#### Install PyTorch with CUDA if available for you machine:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
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

## Datasets

This project uses two publicly available datasets for Portuguese hate speech detection:

1. OffComBR Dataset
   - Source: [OffComBR GitHub Repository](https://github.com/rogersdepelle/OffComBR/tree/master)
   - Description: A dataset of offensive comments in Brazilian Portuguese, collected from news websites and social media.
   - Files used: 
     - OffComBR2.arff
     - OffComBR3.arff

2. HateBR Dataset
   - Source: [HateBR GitHub Repository](https://github.com/franciellevargas/HateBR/tree/main)
   - Description: A large-scale dataset for hate speech detection in Brazilian Portuguese, collected from Instagram.
   - File used:
     - HateBR.csv

These datasets are combined and preprocessed for use in this project. Please refer to the original repositories for more information about the datasets, including their collection methodologies, annotations, and usage terms.

Note: Ensure you comply with the usage terms and provide appropriate attribution when using these datasets.

## Data Preparation

The datasets are preprocessed and combined into two CSV files:
- `dataset/train_val_data.csv`: For training and validation (80% of the combined data)
- `dataset/test_data.csv`: For final evaluation (20% of the combined data)

Each CSV contains 'text' and 'label' columns. The preprocessing steps include:
1. Loading and merging the datasets
2. Splitting into train+validation and test sets
3. Saving as CSV files for easy loading during training and evaluation

## Results

After training, the model will be saved in the specified output path or with a default name based on the model type.

Evaluation results will display accuracy, F1 score, and a detailed classification report.

## Notes

- Ensure you have sufficient GPU resources for training larger models.
- Adjust batch size based on your GPU memory capacity.
- For best results with Portuguese text, the BERTimbau model is recommended.

## Acknowledgements

We would like to thank the creators and contributors of the OffComBR and HateBR datasets for making their data publicly available for research purposes