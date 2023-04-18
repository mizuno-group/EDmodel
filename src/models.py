# 220609

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Config(nn.Module):
    def __init__(
        self,
        vocab_size = 41,
        maxlen = 200,
        gru_layer = [256,512,1024],
        hidden_layer = 256,
        embedding_dim = 128,
        dropout = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.gru_layer = gru_layer
        self.hidden_layer = hidden_layer
        self.embedding_dim = embedding_dim
        self.dropout = dropout


class Encoder(nn.Module):
    def __init__(self,config):
        """
        vocab_size: int, the number of input words
        embedding_dim: int, embedding dimention
        gru_layer: list of int, the size of GRU hidden units
        hidden_layer: int, the unit size of bottleneck layer
        dropout: float [0,1], Dropout ratio
        """
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.gru_layer = config.gru_layer
        self.hidden_layer = config.hidden_layer
        self.dropout = config.dropout

        dims = self.gru_layer.copy()
        dims.insert(0,self.embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim,padding_idx=0)
        self.gru = nn.ModuleList([nn.GRU(dims[i],dims[i+1],1) for i in range(len(self.gru_layer))])
        self.linear = nn.Linear(sum(self.gru_layer),self.hidden_layer)
        self.linear2 = nn.Linear(self.hidden_layer,sum(self.gru_layer))
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self,x,inference=False):
        # x: Tensor, [L,B]
        embedding = self.dropout(self.embedding(x)) # [L,B,E]
        states = []
        for v in self.gru:
            embedding, s = v(embedding) # [L,B,H]
            states.append(s.squeeze(0)) # [B,H]
        states = torch.cat(states,axis=1) # [B,Hsum]
        states = self.linear(states) # [B, Hout]
        if inference == False:
            states = states + torch.normal(0,0.05,size=states.shape).to(DEVICE)
        states = torch.tanh(states)
        states_to_dec = self.linear2(states)
        return embedding, states, states_to_dec


class Decoder(nn.Module):
    def __init__(self,config):
        """
        vocab_size: int, the number of input words
        embedding_dim: int, embedding dimention
        gru_layer: list of int, the size of GRU hidden units
        hidden_layer: int, the unit size of bottleneck layer
        dropout: float [0,1], Dropout ratio
        """
        super().__init__()
        self.gru_layer = config.gru_layer
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.hidden_layer = config.hidden_layer
        dims = self.gru_layer.copy()
        dims.insert(0,self.embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim,padding_idx=0)
        self.gru = nn.ModuleList([nn.GRU(dims[i],dims[i+1],1) for i in range(len(self.gru_layer))])
        self.linear_out = nn.Linear(self.gru_layer[-1],self.vocab_size,bias=False)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self,x,state):
        # x: [L,B]
        # state: [B,H]
        embedding = self.dropout(self.embedding(x)) # [L,B,E]
        state = state.unsqueeze(0) # [1,B,Hsum]
        states = []
        cur_state = 0
        for v,w in zip(self.gru,self.gru_layer):
            embedding, s = v(embedding,state[:,:,cur_state:cur_state+w].contiguous())
            cur_state += w
            states.append(s.squeeze(0))
        output = self.linear_out(embedding)
        return output, torch.cat(states,axis=1)


class Seq2Seq(nn.Module):
    def __init__(self,config):
        """
        vocab_size: int, the number of input words
        embedding_dim: int, embedding dimention
        hidden_dims: list of int, the size of GRU hidden units
        bottleneck_dim: int, the unit size of bottleneck layer
        dropout: float [0,1], Dropout ratio
        """
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self,x,y,inference=False):
        _, latent, todec = self.encoder(x,inference=inference)
        output, _ = self.decoder(y,todec)
        return output, latent

    def encode(self,x,inference=True):
        return self.encoder(x,inference=inference)[1]

    def decode(self,x,state):
        todec = self.encoder.linear2(state)
        return self.decoder(x,todec)


def mish(x):
    return x*torch.tanh(F.softmax(x,dim=-1))
