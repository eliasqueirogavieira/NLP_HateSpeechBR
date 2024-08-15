import torch
from torch import nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class XLMRobertaClassifier(nn.Module):
    def __init__(self, freeze_base=False):
        super(XLMRobertaClassifier, self).__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.classifier = nn.Linear(self.xlm_roberta.config.hidden_size, 2)

        if freeze_base:
            for param in self.xlm_roberta.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        logits = self.classifier(pooled_output)
        return logits


def prepare_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
    return dataset


def train_model(model, train_dataloader, val_dataloader, epochs=3, learning_rate=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

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

        print(
            f'Epoch {epoch + 1}: Avg. Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, F1 Score: {f1:.4f}')

    return model


def process_dataset(merged_df, test_size=0.2, batch_size=32):
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    texts = merged_df['text'].tolist()
    labels = merged_df['label'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=test_size,
                                                                        random_state=42)

    train_dataset = prepare_data(train_texts, train_labels, tokenizer)
    val_dataset = prepare_data(val_texts, val_labels, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader


def main(merged_df, epochs=3, learning_rate=1e-5, batch_size=32):
    model = XLMRobertaClassifier()
    train_dataloader, val_dataloader = process_dataset(merged_df, batch_size=batch_size)
    trained_model = train_model(model, train_dataloader, val_dataloader, epochs=epochs, learning_rate=learning_rate)
    return trained_model
