"""
import torch
import torch.nn as nn
import pickle

from Wav2vec2 import wav2vec2

with open ("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

p2idx = vocab['p2idx']
idx2p = vocab['idx2p']

class canon_embed(nn.Module):
    def __init__(self, phone, emb_dim, hidden_dim=512, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(phone, 64, padding_idx=pad_idx)
        self.bi_lstm = nn.LSTM(
            input_size=64,
            hidden_size=emb_dim//2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.proj  = nn.Linear(emb_dim, hidden_dim)
    def forward(self, canon_input):
        S, _ = self.bi_lstm(self.emb(canon_input))
        S = self.proj(S)
        return S


class pei(nn.Module):
    def __init__(self, phone=len(p2idx), hidden_dim=512):
        super().__init__()
        self.canon_emb = canon_embed(phone=phone, emb_dim=256)
        self.mha1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.mha2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.w2v2_proj = nn.Linear(768, hidden_dim)
        self.ctc_proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, phone)
        )
        self.nll_proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256,2)
        )

    def forward (self, au_input, canon_input, w2v2, input_lengths=None, pad_idx=None):
        X = w2v2(au_input)
        T = X.size(1)
        S = self.canon_emb(canon_input)
        X = self.w2v2_proj(X)

        audio_pad_mask = None
        if input_lengths is not None:
            valid_len = torch.clamp(input_lengths // 320, max=T)
            audio_pad_mask = torch.arange(T, device=X.device).unsqueeze(0) >= valid_len.unsqueeze(1)

        canon_pad_mask = None
        if pad_idx is not None:
            canon_pad_mask = (canon_input == pad_idx)

        error_feature, _ = self.mha1(S, X, X, key_padding_mask=audio_pad_mask)
        Ehs = S - error_feature

        Ec, _ = self.mha2(X, Ehs, Ehs, key_padding_mask=canon_pad_mask)

        ctc_logits = self.ctc_proj(Ec)
        nll_logits = self.nll_proj(Ehs)

        return ctc_logits, nll_logits
"""

import torch
import torch.nn as nn


class canon_embed(nn.Module):
    def __init__(self, phone, emb_dim, hidden_dim=768, pad_idx=68):
        super().__init__()
        self.emb = nn.Embedding(phone+1, 768, padding_idx=pad_idx)
        self.bi_lstm = nn.LSTM(
            input_size=64,
            hidden_size=emb_dim // 2,
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(emb_dim, hidden_dim)

    def forward(self, canon_input):
        S, _ = self.bi_lstm(self.emb(canon_input))
        S = self.proj(S)
        return S


class pei(nn.Module):

    def __init__(self, phone, pad_idx=68, hidden_dim=768):
        super().__init__()
        self.canon_emb = nn.Embedding(phone, 768, padding_idx=pad_idx)
        self.mha1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=16, batch_first=True, dropout=0.2)
        self.mha2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=16, batch_first=True, dropout=0.2)
        self.ctc_proj = nn.Linear(768,phone)
        self.nll_proj = nn.Linear(768,2)

    def forward(self, au_input, canon_input, w2v2):
        X = w2v2(au_input)
        T = X.size(1)
        S = self.canon_emb(canon_input)

        error_feature, _ = self.mha1(S, X, X)
        Ehs = S - error_feature

        Ec, _ = self.mha2(X, Ehs, Ehs)

        ctc_logits = self.ctc_proj(Ec)
        nll_logits = self.nll_proj(Ehs)

        return ctc_logits, nll_logits