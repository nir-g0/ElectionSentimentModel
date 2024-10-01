import torch  # Import PyTorch library
from transformers import BertTokenizer, BertForSequenceClassification  # Import BERT-related modules from transformers
from torch.utils.data import DataLoader, TensorDataset  # Import data handling utilities from PyTorch
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
from sklearn.model_selection import train_test_split  # Import function for splitting dataset
import random  # Import random module (not used in this code)

# Load and Preprocess Data:
df = pd.read_csv('tweets.csv', encoding='ISO-8859-1')  # Read CSV file containing tweets
texts = df.iloc[:,-1].tolist()  # Extract texts from the last column
labels = df.iloc[:,0].tolist()  # Extract labels from the first column
print('Read Dataset')  # Print confirmation message

# Convert labels to integers
label_dict = {0: 0, 4: 1}  # Define dictionary to map original labels to binary labels
labels = [label_dict[label] for label in labels]  # Convert labels using the dictionary

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)  # Split data into training and validation sets

# Load BERT Tokenizer and Model:
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Initialize BERT tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Initialize BERT model for sequence classification

print('Encoding data...')  # Print status message

# Tokenize and encode
def encode_data(texts, labels, tokenizer, max_length=128):
    input_ids = []  # Initialize list to store tokenized input ids
    attention_masks = []  # Initialize list to store attention masks
    shortened_labels = []  # Initialize list to store labels for processed texts
    for i in range(0, len(texts), 10000):  # Process texts in batches of 10000
        encoded = tokenizer.encode_plus(  # Encode a single text
            texts[i],
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        shortened_labels.append(labels[i])  # Add corresponding label
        input_ids.append(encoded['input_ids'])  # Add tokenized input to list
        attention_masks.append(encoded['attention_mask'])  # Add attention mask to list
    
    input_ids = torch.cat(input_ids, dim=0)  # Concatenate all tokenized inputs
    attention_masks = torch.cat(attention_masks, dim=0)  # Concatenate all attention masks
    labels = torch.tensor(shortened_labels)  # Convert labels to PyTorch tensor
    
    return input_ids, attention_masks, labels

train_input_ids, train_attention_masks, train_labels = encode_data(train_texts, train_labels, tokenizer)  # Encode training data
val_input_ids, val_attention_masks, val_labels = encode_data(val_texts, val_labels, tokenizer)  # Encode validation data
print('Data Encoded.')  # Print confirmation message

# Create DataLoaders:
print('Creating Data Loaders...')  # Print status message
batch_size = 32  # Set batch size
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)  # Create training dataset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Create training dataloader

val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)  # Create validation dataset
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Create validation dataloader
print('Data Loaders created.')  # Print confirmation message

# Training parameters:
print('Training Parameters...')  # Print status message
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device (GPU if available, else CPU)
model.to(device)  # Move model to the selected device

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # Initialize AdamW optimizer
epochs = 8  # Set number of training epochs

for epoch in range(epochs):  # Loop through epochs
    print(f"Epoch #{epoch}")  # Print current epoch number
    model.train()  # Set model to training mode
    for batch in train_dataloader:  # Iterate through batches
        batch = tuple(t.to(device) for t in batch)  # Move batch to device
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}  # Prepare inputs
        
        outputs = model(**inputs)  # Forward pass
        loss = outputs.loss  # Get loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Reset gradients
print('Model Trained')  # Print confirmation message

# Validation
print('Evaluating Model...')  # Print status message
model.eval()  # Set model to evaluation mode
val_accuracy = 0  # Initialize validation accuracy
for batch in val_dataloader:  # Iterate through validation batches
    batch = tuple(t.to(device) for t in batch)  # Move batch to device
    inputs = {'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]}  # Prepare inputs
    
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)  # Forward pass
    
    logits = outputs.logits  # Get logits
    predictions = torch.argmax(logits, dim=-1)  # Get predictions
    val_accuracy += (predictions == inputs['labels']).float().mean()  # Calculate accuracy

val_accuracy /= len(val_dataloader)  # Calculate average validation accuracy
print(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {val_accuracy:.4f}")  # Print validation accuracy

def predict_sentiment(text):  # Define function to predict sentiment
    encoded = tokenizer.encode_plus(  # Encode input text
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    input_ids = encoded['input_ids'].to(device)  # Move input ids to device
    attention_mask = encoded['attention_mask'].to(device)  # Move attention mask to device
    
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(input_ids, attention_mask=attention_mask)  # Forward pass
    
    logits = outputs.logits  # Get logits
    prediction = torch.argmax(logits, dim=-1)  # Get prediction
    return 'positive' if prediction.item() == 1 else 'negative'  # Return sentiment label

# Example usage
text = "I love this movie! It's amazing!"  # Example input text
sentiment = predict_sentiment(text)  # Predict sentiment
print(f"Sentiment: {sentiment}")  # Print predicted sentiment