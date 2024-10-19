import torch
import torch.nn as nn
import math
from tqdm import tqdm
from encoder import Encoder,EncoderBlock
from decoder import Decoder,DecoderBlock,ProjectionLayer
from utils import InputEmbeddings,PositionalEncoding,MultiHeadAttentionBlock,FeedForwardBlock


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, embedding_dim)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, embedding_dim)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    



def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, embedding_dim: int=512, N: int=6, h: int=8, dropout: float=0.1, hidden_dim: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(embedding_dim, src_vocab_size)
    tgt_embed = InputEmbeddings(embedding_dim, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(embedding_dim, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(embedding_dim, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(embedding_dim, h, dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim, hidden_dim, dropout)
        encoder_block = EncoderBlock(embedding_dim, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(embedding_dim, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(embedding_dim, h, dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim, hidden_dim, dropout)
        decoder_block = DecoderBlock(embedding_dim, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(embedding_dim, nn.ModuleList(encoder_blocks))
    decoder = Decoder(embedding_dim, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(embedding_dim, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer