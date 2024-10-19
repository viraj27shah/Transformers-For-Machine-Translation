import torch
import torch.nn as nn
import math
from tqdm import tqdm

class InputEmbeddings(nn.Module):
    def __init__(self, embedding_dim, vocab_size) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, embedding_dim)
        return self.embedding(x) * math.sqrt(self.embedding_dim)
    

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_len, dropout) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # matrix of shape seq_len x embedding_dim
        pe = torch.zeros(seq_len, embedding_dim)
        # vector of shape seq_len x 1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # vector of shape embedding_dim x 1
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / embedding_dim))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / embedding_dim))
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, embedding_dim)
        
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, embedding_dim)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    # suppose you have 3 items in batch then it will pick each item and find mean and svd and normalize own value.
    # also learn alpha(multiplicative) and bias(additive) to add fluctuations(amplify the value) in data when necessary. It will learn this.
    def __init__(self, features, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        # Making the parameter learnable
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # (batch, seq_len, 1) output dimension of below line
        # dim = -1 to apply mean on last dimension
        # generally mean will remove dimension, so we are keeping it
        mean = x.mean(dim = -1, keepdim = True) 
        std = x.std(dim = -1, keepdim = True)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class FeedForwardBlock(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear_layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # (batch, seq_len, embedding_dim) --> (batch, seq_len, hidden_dim) --> (batch, seq_len, embedding_dim)
        return self.linear_layer2(self.dropout(torch.relu(self.linear_layer1(x))))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, embedding_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        # no of heads
        self.heads = heads
        
        # Dimension of vector seen by each head
        self.d_k = embedding_dim // heads
        self.w_q = nn.Linear(embedding_dim, embedding_dim, bias=False) # Wq
        self.w_k = nn.Linear(embedding_dim, embedding_dim, bias=False) # Wk
        self.w_v = nn.Linear(embedding_dim, embedding_dim, bias=False) # Wv
        
        # after concatnating all heads we just take to desired dimension
        self.w_o = nn.Linear(embedding_dim, embedding_dim, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    #Attention score calculation
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask 
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, embedding_dim) --> (batch, seq_len, embedding_dim)
        key = self.w_k(k) # (batch, seq_len, embedding_dim) --> (batch, seq_len, embedding_dim)
        value = self.w_v(v) # (batch, seq_len, embedding_dim) --> (batch, seq_len, embedding_dim)

        # (batch, seq_len, embedding_dim) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, embedding_dim)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, embedding_dim) --> (batch, seq_len, embedding_dim)  
        return self.w_o(x)