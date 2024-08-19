import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False, *args, **kwargs):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def prepare_data(texts, labels, tokenizer, max_length=128):
    # Ensure texts and labels are lists
    if not isinstance(texts, list) or not isinstance(labels, list):
        raise ValueError("Both 'texts' and 'labels' must be lists.")

    # Ensure texts and labels have the same length
    if len(texts) != len(labels):
        raise ValueError(f"Number of texts ({len(texts)}) does not match number of labels ({len(labels)}).")

    # Remove any None or empty string entries
    valid_data = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        if text and isinstance(text, str):
            valid_data.append((text, label))
        else:
            print(f"Invalid entry at index {i}:")
            print(f"Text: {repr(text)}")
            print(f"Label: {label}")
            print(f"Type of text: {type(text)}")
            print("---")

    if len(valid_data) != len(texts):
        print(f"Warning: Removed {len(texts) - len(valid_data)} invalid entries.")

    texts, labels = zip(*valid_data)

    try:
        encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    except Exception as e:
        print(f"Error during tokenization: {e}")
        print(f"First few texts: {texts[:5]}")
        raise

    try:
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
    except Exception as e:
        print(f"Error creating TensorDataset: {e}")
        print(f"Shape of input_ids: {encodings['input_ids'].shape}")
        print(f"Shape of attention_mask: {encodings['attention_mask'].shape}")
        print(f"Length of labels: {len(labels)}")
        raise

    return dataset


def train_model(model, train_dataloader, val_dataloader=None, epochs=3, learning_rate=2e-5, patience=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'\nEpoch {epoch + 1}: Avg. Training Loss: {avg_train_loss:.4f}')

        if val_dataloader is not None:
            model.eval()
            val_accuracy = 0
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    outputs = model(input_ids, attention_mask)
                    predictions = torch.argmax(outputs, dim=1)
                    val_accuracy += (predictions == labels).sum().item()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_accuracy /= len(val_dataloader.dataset)
            f1 = f1_score(all_labels, all_predictions, average='weighted')

            print(f'Validation Accuracy: {val_accuracy:.4f}, F1 Score: {f1:.4f}')

            # Early stopping logic
            if f1 > best_f1:
                best_f1 = f1
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

    # Load the best model if early stopping occurred
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, best_f1


def process_dataset(merged_df, test_size=0.2, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    texts = merged_df['text'].tolist()
    labels = merged_df['label'].tolist()

    if test_size > 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=test_size,
                                                                            random_state=42)
        train_dataset = prepare_data(train_texts, train_labels, tokenizer)
        val_dataset = prepare_data(val_texts, val_labels, tokenizer)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader
    else:
        dataset = prepare_data(texts, labels, tokenizer)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, None
