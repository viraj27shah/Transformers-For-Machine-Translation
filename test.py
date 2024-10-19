import torch
import torch.nn as nn
import math
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
import re
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

from build_transformer import build_transformer
# from train import generate_square_subsequent_mask,train_epoch,evaluate
# from train import TranslationDataset,TranslationDatasetForTest


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nltk.download('punkt')

def tokenize(text):
    # Remove punctuation using regex, then find words
    return re.findall(r'\b\w+\b', text.lower())

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_lang='en', tgt_lang='fr', max_len=50):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.max_len = max_len
        
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = f.readlines()
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = f.readlines()

        self.src_vocab,self.src_id2word = self.build_vocab(self.src_sentences)
        self.tgt_vocab,self.tgt_id2word = self.build_vocab(self.tgt_sentences)
        
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)
    
    
        
    def build_vocab(self, sentences):
        # Tokenize sentences into words and lower case
#         tokens = [word_tokenize(sentence.lower()) for sentence in sentences]
        tokens = [tokenize(sentence.lower()) for sentence in sentences]


        # Create word frequency counter
        vocab = Counter([token for sentence in tokens for token in sentence])

        # Create word to index mapping, reserving special tokens
        word2id = {word: idx+4 for idx, (word, _) in enumerate(vocab.items())}  # Reserve 0: PAD, 1: UNK, 2: BOS, 3: EOS
        word2id['<PAD>'] = 0
        word2id['<UNK>'] = 1
        word2id['<SOS>'] = 2
        word2id['<EOS>'] = 3

        # Create index to word mapping by reversing word2id
        id2word = {idx: word for word, idx in word2id.items()}

        return word2id, id2word

    def encode_sentence(self, sentence, vocab):
        tokens = word_tokenize(sentence.lower())
        tokens = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        tokens = [vocab['<SOS>']] + tokens  # Add SOS token at the start

        # Ensure EOS is added before trimming and padding
        tokens.append(vocab['<EOS>'])

        # Trim the tokens to max_len - 1 to leave space for EOS
        tokens = tokens[:self.max_len - 1]

        # Ensure EOS is at the end before padding
        if tokens[-1] != vocab['<EOS>']:
            tokens.append(vocab['<EOS>'])

        # Pad the remaining tokens if necessary
        tokens += [vocab['<PAD>']] * (self.max_len - len(tokens))

        return tokens

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        src_encoded = self.encode_sentence(src_sentence, self.src_vocab)
        tgt_encoded = self.encode_sentence(tgt_sentence, self.tgt_vocab)
        
        return torch.tensor(src_encoded), torch.tensor(tgt_encoded)
    
class TranslationDatasetForTest(Dataset):
    def __init__(self, src_file, tgt_file,src_vocab,tgt_vocab,src_lang='en', tgt_lang='fr', max_len=50):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.max_len = max_len
        
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = f.readlines()
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = f.readlines()

#         self.src_vocab,self.src_id2word = self.build_vocab(self.src_sentences)
#         self.tgt_vocab,self.tgt_id2word = self.build_vocab(self.tgt_sentences)
        
#         self.src_vocab_size = len(self.src_vocab)
#         self.tgt_vocab_size = len(self.tgt_vocab)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def encode_sentence(self, sentence, vocab):
        tokens = word_tokenize(sentence.lower())
        tokens = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        tokens = [vocab['<SOS>']] + tokens  # Add SOS token at the start

        # Ensure EOS is added before trimming and padding
        tokens.append(vocab['<EOS>'])

        # Trim the tokens to max_len - 1 to leave space for EOS
        tokens = tokens[:self.max_len - 1]

        # Ensure EOS is at the end before padding
        if tokens[-1] != vocab['<EOS>']:
            tokens.append(vocab['<EOS>'])

        # Pad the remaining tokens if necessary
        tokens += [vocab['<PAD>']] * (self.max_len - len(tokens))

        return tokens


    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        src_encoded = self.encode_sentence(src_sentence, self.src_vocab)
        tgt_encoded = self.encode_sentence(tgt_sentence, self.tgt_vocab)
        
        return torch.tensor(src_encoded), torch.tensor(tgt_encoded)


def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence to mask future positions."""
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train_epoch(model, dataloader, optimizer, criterion, src_pad_idx, tgt_pad_idx, device):
    model.train()
    epoch_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)

        # Prepare inputs for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
#         tgt_mask = (tgt_input != tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        # Size of the target sequence (i.e., sequence length)
        seq_len = tgt_input.size(1)

        # Padding mask: mask padded tokens
        padding_mask = (tgt_input != tgt_pad_idx).unsqueeze(1).unsqueeze(2).float()  # Convert to float

        # Look-ahead mask: mask future tokens
        look_ahead_mask = generate_square_subsequent_mask(seq_len).to(device)  # [seq_len, seq_len]

        # Combine padding mask and look-ahead mask by multiplication
        tgt_mask = padding_mask * look_ahead_mask.unsqueeze(0)


        optimizer.zero_grad()

        # Forward pass
        encoder_output = model.encode(src, src_mask)
        output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
        output = model.project(output)

        # Compute loss
        output = output.view(-1, output.shape[-1])
        tgt_output = tgt_output.contiguous().view(-1)
        loss = criterion(output, tgt_output)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)



def tensor_to_sentence(tensor, vocab):
    """ Converts a tensor of token IDs into a sentence using the vocab (id2word mapping). """
#     print("SAFNS")
#     print(tensor)
#     print(vocab)
    words = [vocab[token] for token in tensor if token in vocab and vocab[token] not in ['<PAD>', '<SOS>', '<EOS>','<UNK>']]
#     print("herer")
#     print(words)
    sentence = ' '.join(words)
    return sentence

def evaluate(model, dataloader, criterion, src_pad_idx, tgt_pad_idx, device, tgt_vocab):
    model.eval()
    epoch_loss = 0
    total_bleu = 0
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)

            # Prepare inputs for teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
#             tgt_mask = (tgt_input != tgt_pad_idx).unsqueeze(1).unsqueeze(2)

            # Size of the target sequence (i.e., sequence length)
            seq_len = tgt_input.size(1)

            # Padding mask: mask padded tokens
            padding_mask = (tgt_input != tgt_pad_idx).unsqueeze(1).unsqueeze(2).float()  # Convert to float

            # Look-ahead mask: mask future tokens
            look_ahead_mask = generate_square_subsequent_mask(seq_len).to(device)  # [seq_len, seq_len]

            # Combine padding mask and look-ahead mask by multiplication
            tgt_mask = padding_mask * look_ahead_mask.unsqueeze(0)


            # Forward pass
            encoder_output = model.encode(src, src_mask)
            output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
            output = model.project(output)

            # Compute loss
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.contiguous().view(-1)
            loss = criterion(output, tgt_output)

            # Accumulate the loss
            epoch_loss += loss.item()

            # Calculate BLEU score
            output = output.view(tgt.size(0), tgt.size(1) - 1, -1)  # Reshape to [batch_size, seq_len, vocab_size]
            pred_tokens = output.argmax(-1).cpu().tolist()  # Get the predicted token indices

            # Convert predictions and targets to sentences
#             print(tgt)
#             print(pred_tokens)
#             cnt = 0
            for i in range(tgt.size(0)):
#                 print()
                pred_sentence = tensor_to_sentence(pred_tokens[i], tgt_id2word)
                ref_sentence = tensor_to_sentence(tgt[i, 1:].cpu().tolist(), tgt_id2word)  # tgt[i, 1:] skips <SOS>
                src_sentence = tensor_to_sentence(src[i, 1:].cpu().tolist(), src_id2word)
#                 print("ss ",src_sentence)
#                 print("ps ",pred_sentence)
#                 print("rs ",ref_sentence)
#                 print()
                
                
                # Split sentences into tokens (list of words)
                pred_tokens_split = pred_sentence.split()
                ref_tokens_split = [ref_sentence.split()]  # BLEU expects a list of reference token lists
                
                # Compute BLEU score for this sentence pair
                bleu_score = sentence_bleu(ref_tokens_split, pred_tokens_split, weights=(0.25, 0.25, 0.25, 0.25))
                total_bleu += bleu_score
#                 cnt += 1
                
#                 if cnt == 5:
#                     break
#             break

    avg_bleu = total_bleu / len(dataloader.dataset)  # Average BLEU score over the dataset
    return epoch_loss / len(dataloader), avg_bleu


# Initialize model
embedding_dim = 512
N = 6  # Number of layers in the encoder/decoder
h = 8  # Number of heads in the attention mechanism
hidden_dim = 2048
dropout = 0.1
epochs = 4
lr = 0.001
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Dataset and DataLoader
train_dataset = TranslationDataset('train.en', 'train.fr')

src_vocab = train_dataset.src_vocab
tgt_vocab = train_dataset.tgt_vocab
src_id2word = train_dataset.src_id2word
tgt_id2word = train_dataset.tgt_id2word

val_dataset = TranslationDatasetForTest('dev.en', 'dev.fr',src_vocab,tgt_vocab)
test_dataset = TranslationDatasetForTest('test.en', 'test.fr',src_vocab,tgt_vocab)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Build transformer
src_vocab_size = len(train_dataset.src_vocab)
tgt_vocab_size = len(train_dataset.tgt_vocab)
# src_id2word = len(train_dataset.src_id2word)
# tgt_id2word = len(train_dataset.tgt_id2word)
src_seq_len = train_dataset.max_len
tgt_seq_len = train_dataset.max_len

file = "transformer.pt"
model = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, embedding_dim, N, h, dropout, hidden_dim)
state_dict = torch.load(file, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model = model.to(device)

print("model loaded")


optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])

test_loss,avg_bleu = evaluate(model, test_loader, criterion, src_vocab['<PAD>'], tgt_vocab['<PAD>'], device,tgt_vocab)

print(f"Test Loss: {test_loss:.4f} , {avg_bleu}")


def tensor_to_sentence_filewrite(tensor, vocab):
    """ Converts a tensor of token IDs into a sentence using the vocab (id2word mapping). """
    words = [vocab[token] for token in tensor if token in vocab and vocab[token] not in ['<PAD>', '<SOS>', '<EOS>','<UNK>']]
    sentence = ' '.join(words)
    return sentence

def evaluate_filewrite(model, dataloader, criterion, src_pad_idx, tgt_pad_idx, device, tgt_vocab, output_file='testbleu.txt'):
    model.eval()
    epoch_loss = 0
    total_bleu = 0

    # Open the file to write the sentences and BLEU scores
    with open(output_file, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for src, tgt in tqdm(dataloader, desc="Evaluating"):
                src, tgt = src.to(device), tgt.to(device)

                # Prepare inputs for teacher forcing
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
#                 tgt_mask = (tgt_input != tgt_pad_idx).unsqueeze(1).unsqueeze(2)

                # Size of the target sequence (i.e., sequence length)
                seq_len = tgt_input.size(1)

                # Padding mask: mask padded tokens
                padding_mask = (tgt_input != tgt_pad_idx).unsqueeze(1).unsqueeze(2).float()  # Convert to float

                # Look-ahead mask: mask future tokens
                look_ahead_mask = generate_square_subsequent_mask(seq_len).to(device)  # [seq_len, seq_len]

                # Combine padding mask and look-ahead mask by multiplication
                tgt_mask = padding_mask * look_ahead_mask.unsqueeze(0)


                # Forward pass
                encoder_output = model.encode(src, src_mask)
                output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
                output = model.project(output)

                # Compute loss
                output = output.view(-1, output.shape[-1])
                tgt_output = tgt_output.contiguous().view(-1)
                loss = criterion(output, tgt_output)

                # Accumulate the loss
                epoch_loss += loss.item()

                # Calculate BLEU score
                output = output.view(tgt.size(0), tgt.size(1) - 1, -1)  # Reshape to [batch_size, seq_len, vocab_size]
                pred_tokens = output.argmax(-1).cpu().tolist()  # Get the predicted token indices

                # Convert predictions and targets to sentences
                for i in range(tgt.size(0)):  # Since batch size is 1, it will process 1 sentence at a time
                    pred_sentence = tensor_to_sentence_filewrite(pred_tokens[i], tgt_id2word)
                    ref_sentence = tensor_to_sentence_filewrite(tgt[i, 1:].cpu().tolist(), tgt_id2word)  # Skip <SOS>
                    src_sentence = tensor_to_sentence_filewrite(src[i, 1:].cpu().tolist(), src_id2word)

                    # Split sentences into tokens (list of words)
                    pred_tokens_split = pred_sentence.split()
                    ref_tokens_split = [ref_sentence.split()]  # BLEU expects a list of reference token lists

                    # Compute BLEU score for this sentence pair
                    bleu_score = sentence_bleu(ref_tokens_split, pred_tokens_split, weights=(0.25, 0.25, 0.25, 0.25))
                    total_bleu += bleu_score

                    # Write the source, predicted, reference sentences, and BLEU score to file
#                     f.write(f"Soource: {src_sentence}\t")
                    f.write(f"{ref_sentence}\t")
#                     f.write(f"Predicted: {pred_sentence}\t")
                    f.write(f"BLEU Score: {bleu_score:.4f}\n")
                    f.write("\n")  # Add a blank line for separation

    avg_bleu = total_bleu / len(dataloader.dataset)  # Average BLEU score over the dataset
    return epoch_loss / len(dataloader), avg_bleu


file = "transformer.pt"
loaded_model = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, embedding_dim, N, h, dropout, hidden_dim)
state_dict = torch.load(file, map_location=torch.device('cpu'))
loaded_model.load_state_dict(state_dict)
loaded_model = loaded_model.to(device)

optimizer = optim.Adam(loaded_model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])

test_loss,avg_bleu = evaluate_filewrite(loaded_model, test_loader, criterion, src_vocab['<PAD>'], tgt_vocab['<PAD>'], device,tgt_vocab)

print(f"Test Loss: {test_loss:.4f} , {avg_bleu}")


# Update the main train_and_evaluate function to record history
def train_and_evaluate_hyper_parameter(model, train_loader, val_loader, test_loader, src_vocab, tgt_vocab, src_pad_idx, tgt_pad_idx, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    
    history = {'train_loss': [], 'val_loss': [], 'bleu': []}
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, src_pad_idx, tgt_pad_idx, device)
        history['train_loss'].append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation
        val_loss, bleu = evaluate(model, val_loader, criterion, src_pad_idx, tgt_pad_idx, device, tgt_vocab)
        history['val_loss'].append(val_loss)
        history['bleu'].append(bleu)
        print(f"Validation Loss: {val_loss:.4f}, BLEU: {bleu:.4f}")

    # Testing
    test_loss, bleu = evaluate(model, test_loader, criterion, src_pad_idx, tgt_pad_idx, device, tgt_vocab)
    print(f"Test Loss: {test_loss:.4f}, BLEU: {bleu:.4f}")

    return history

# Define a function for hyperparameter tuning
def hyperparameter_tuning(train_loader, val_loader, test_loader, src_vocab, tgt_vocab, src_pad_idx, tgt_pad_idx, device, epochs=4):
    # Hyperparameter combinations to try
    hyperparams = [
        {'layers': 4, 'heads': 8, 'embedding_dim': 256, 'hidden_dim': 1024, 'dropout': 0.1},
        {'layers': 6, 'heads': 8, 'embedding_dim': 512, 'hidden_dim': 2048, 'dropout': 0.2},
        {'layers': 8, 'heads': 16, 'embedding_dim': 512, 'hidden_dim': 2048, 'dropout': 0.1},
        {'layers': 6, 'heads': 8, 'embedding_dim': 256, 'hidden_dim': 1024, 'dropout': 0.2},
    ]
    
    results = []

    for idx, params in enumerate(hyperparams):
        print(f"Training with combination {idx + 1}: {params}")

        # Build model with current hyperparameters
        model = build_transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            src_seq_len=train_dataset.max_len,
            tgt_seq_len=train_dataset.max_len,
            embedding_dim=params['embedding_dim'],
            N=params['layers'],
            h=params['heads'],
            dropout=params['dropout'],
            hidden_dim=params['hidden_dim']
        )
        model = model.to(device)

        # Train and evaluate model
        history = train_and_evaluate_hyper_parameter(model, train_loader, val_loader, test_loader,
                                     src_vocab, tgt_vocab,
                                     src_pad_idx, tgt_pad_idx,
                                     epochs=epochs, lr=0.001, device=device)
        
        # Store results
        results.append({'params': params, 'history': history})

    return results

# Plotting function to visualize loss and BLEU score
def plot_training_history(results):
    for result in results:
        params = result['params']
        history = result['history']
        
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss', color='blue')
        plt.plot(history['val_loss'], label='Val Loss', color='orange')
        plt.title(f"Loss for {params}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()
        
        # Plot BLEU score
        plt.figure(figsize=(10, 5))
        plt.plot(history['bleu'], label='BLEU Score', color='green')
        plt.title(f"BLEU Score for {params}")
        plt.xlabel("Epochs")
        plt.ylabel("BLEU Score")
        plt.legend()
        plt.grid()
        plt.show()



# Execute hyperparameter tuning and plot results
results = hyperparameter_tuning(train_loader, val_loader, test_loader, src_vocab, tgt_vocab, 
                                src_pad_idx=src_vocab['<PAD>'], tgt_pad_idx=tgt_vocab['<PAD>'], 
                                device=device)

# Plot the results of hyperparameter tuning
plot_training_history(results)