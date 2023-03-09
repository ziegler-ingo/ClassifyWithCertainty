"""
Transformer implementations are based on:
    Sinkformers: Transformers with Doubly Stochastic Attention
    https://arxiv.org/abs/2110.11773
    https://github.com/michaelsdr/sinkformers

    Transformer Uncertainty Estimation with Hierarchical Stochastic Attention
    https://arxiv.org/abs/2112.13776
    https://github.com/amzn/sto-transformer
"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from sinkhorn import SinkhornDistance


class SelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        heads,
        kind=None,
        spectral=True,
        tau1=None,
        tau2=None,
        k_centroid=None,
        init_fn=nn.init.uniform_,
    ):
        super().__init__()

        self.kind = kind
        self.emb_dim = emb_dim
        self.heads = heads
        self.head_dim = emb_dim // heads

        if kind == "sto":
            assert tau2 is not None, "kind = 'sto' requires tau2 to be defined."
            self.tau2 = tau2
        elif kind == "sto_dual":
            assert (
                tau1 is not None and tau2 is not None and k_centroid is not None
            ), "kind = 'sto_dual' requires tau1, tau2, k_centroid to be defined."
            self.tau1 = tau1
            self.tau2 = tau2
            self.centroid = nn.Parameter(
                init_fn(torch.empty(self.head_dim, k_centroid), a=-0.5, b=0.5)
            )

        if spectral:
            self.Q_linear = spectral_norm(nn.Linear(self.head_dim, self.head_dim))
            self.K_linear = spectral_norm(nn.Linear(self.head_dim, self.head_dim))
            self.V_linear = spectral_norm(nn.Linear(self.head_dim, self.head_dim))
            self.FC_linear = spectral_norm(nn.Linear(heads * self.head_dim, emb_dim))
        else:
            self.Q_linear = nn.Linear(self.head_dim, self.head_dim)
            self.K_linear = nn.Linear(self.head_dim, self.head_dim)
            self.V_linear = nn.Linear(self.head_dim, self.head_dim)
            self.FC_linear = nn.Linear(heads * self.head_dim, emb_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # input shapes: (batch_size, seq_len, emb_dim)
        # after view: (batch_size, seq_len, heads, head_dim)
        Q = self.Q_linear(query.view(batch_size, -1, self.heads, self.head_dim))
        K = self.K_linear(key.view(batch_size, -1, self.heads, self.head_dim))
        V = self.V_linear(value.view(batch_size, -1, self.heads, self.head_dim))

        if self.kind == "sto_dual":
            # einsum(nshd, dc -> nshc), nshc has (batch_size, seq_len, heads, k_centroid)
            # we sum over head_dim, i.e. we need head_dim once at previous to last, once at last dim
            K_ = torch.einsum("nshd, dc -> nshc", K, self.centroid)
            prob = F.gumbel_softmax(K_, tau=self.tau1, hard=False, dim=-1)
            # sto_K shape out: (batch_size, seq_len, heads, head_dim)
            sto_K = torch.einsum("nshc, cd -> nshd", prob, self.centroid.T)
            # key_out shape out: (batch_size, heads, seq_len, seq_len)
            key_out = torch.einsum("nqhd, nkhd -> nhqk", Q, sto_K)
        else:
            # einsum(nqhd, nkhd -> nhqk), nhqk has (batch_size, heads, seq_len, seq_len)
            # we sum over head_dim, i.e. we need head_dim once at previous to last, once at last dim
            key_out = torch.einsum("nqhd, nkhd -> nhqk", Q, K)

        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, -1e20)

        if self.kind == "sinkhorn":
            sinkhorn = SinkhornDistance(eps=1, max_iter=3)
            attn = sinkhorn(
                key_out.view(-1, key_out.shape[2], key_out.shape[3])
                / (self.emb_dim ** (1 / 2))
            )[0]
            attn = attn * attn.shape[-1]
            attn = attn.view(key_out.shape)
        elif self.kind == "sto" or self.kind == "sto_dual":
            attn = F.gumbel_softmax(key_out, tau=self.tau2, hard=False, dim=3)
        else:
            attn = torch.softmax(key_out / (self.emb_dim ** (1 / 2)), dim=3)

        # attention: (batch_size, heads, seq_len, seq_len)
        # V: (batch_size, seq_len, heads, head_dim)
        # einsum_out: (batch_size, seq_len, heads, head_dim)
        # reshape out: (batch_size, seq_len, emb_dim)
        out = torch.einsum("nhql, nlhd -> nqhd", attn, V).reshape(
            batch_size, query.shape[1], self.heads * self.head_dim
        )

        return self.FC_linear(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        n_heads,
        dropout,
        forward_dim,
        spectral=True,
        kind=None,
        tau1=None,
        tau2=None,
        k_centroid=None,
    ):
        super().__init__()

        if kind == "sto":
            assert tau2 is not None, "kind = 'sto' requires tau2 to be defined."
            self.mha = SelfAttention(
                emb_dim, n_heads, kind="sto", spectral=spectral, tau2=tau2
            )
        elif kind == "sto_dual":
            assert (
                tau1 is not None and tau2 is not None and k_centroid is not None
            ), "kind = 'sto_dual' requires tau1, tau2, k_centroid to be defined."
            self.mha = SelfAttention(
                emb_dim,
                n_heads,
                kind="sto_dual",
                spectral=spectral,
                tau1=tau1,
                tau2=tau2,
                k_centroid=k_centroid,
            )
        elif kind == "sinkhorn":
            self.mha = SelfAttention(emb_dim, n_heads, kind="sinkhorn")
        else:
            self.mha = SelfAttention(emb_dim, n_heads)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-6)

        if spectral:
            self.ffn = nn.Sequential(
                spectral_norm(nn.Linear(emb_dim, forward_dim)),
                nn.ReLU(),
                spectral_norm(nn.Linear(forward_dim, emb_dim)),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(emb_dim, forward_dim),
                nn.ReLU(),
                nn.Linear(forward_dim, emb_dim),
            )

    def forward(self, query, key, value, mask):
        # q, k, v: (batch_size, seq_len, emb_dim)
        # mask: (batch_size, 1, 1, seq_len)
        attention = self.mha(query, key, value, mask)

        # skip connection --> norm --> dropout
        # input shape: (batch_size, seq_len, emb_dim)
        x = self.dropout(self.norm1(attention + query))

        # input shape: (batch_size, seq_len, emb_dim)
        # output shape: (batch_size, seq_len, emb_dim)
        ffn = self.ffn(x)
        out = self.dropout(self.norm2(ffn + x))
        return out


class EncoderNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        n_layers,
        n_heads,
        forward_dim,
        dropout,
        max_len,
        kind=None,
        spectral=True,
        tau1=None,
        tau2=None,
        k_centroid=None,
        n_classes=2,
        device="cpu",
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.kind = kind
        self.device = device

        if kind == "sinkhorn":
            self.sinusoid_table = get_sinusoid_table(max_len + 1, emb_dim)
            self.pos_embedding = nn.Embedding.from_pretrained(
                self.sinusoid_table, freeze=True
            )
        else:
            self.pos_embedding = nn.Embedding(max_len, emb_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim,
                    n_heads,
                    dropout,
                    forward_dim,
                    kind=kind,
                    spectral=spectral,
                    tau1=tau1,
                    tau2=tau2,
                    k_centroid=k_centroid,
                )
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(emb_dim, n_classes)

    def forward(self, x, mask, return_embeddings=None, emb=None):
        batch_size, seq_len = x.shape

        if self.kind == "sinkhorn":
            positions = (torch.arange(seq_len).expand(batch_size, seq_len) + 1).to(
                self.device
            )
        else:
            positions = (
                torch.arange(seq_len).expand(batch_size, seq_len).to(self.device)
            )

        sum_emb = self.embedding(x) + self.pos_embedding(positions)
        if emb is not None:
            sum_emb = sum_emb + emb
        out = self.dropout(sum_emb)

        for layer in self.layers:
            out = layer(out, out, out, mask)
        if self.kind == "sinkhorn":
            out = out.max(dim=1)[0]
        else:
            out = out[:, 0, :]  # use CLS token for classification

        if return_embeddings is not None:
            return out
        return self.fc_out(out)


class TransformerEncoder(nn.Module):
    """
    Build a Transformer Encoder.

    Arguments
    _________
    kind : Defines the type of attention used. Options: sinkhorn, sto, sto_dual.
        If empty or None, standard scaled dot-product attention is used.
    """

    def __init__(
        self,
        vocab_size,
        emb_dim,
        n_layers,
        n_heads,
        forward_dim,
        dropout,
        max_len,
        pad_idx,
        kind=None,
        spectral=True,
        tau1=None,
        tau2=None,
        k_centroid=None,
        n_classes=2,
        device="cpu",
    ):
        super().__init__()

        self.encoder = EncoderNetwork(
            vocab_size,
            emb_dim,
            n_layers,
            n_heads,
            forward_dim,
            dropout,
            max_len,
            kind=kind,
            spectral=spectral,
            tau1=tau1,
            tau2=tau2,
            k_centroid=k_centroid,
            n_classes=n_classes,
            device=device,
        )

        self.pad_idx = pad_idx
        self.device = device

    def make_mask(self, seq):
        # creates mask shape: (batch_size, 1, 1, seq_len)
        mask = (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask.to(self.device)

    def forward(self, seq, return_embeddings=None, emb=None):
        mask = self.make_mask(seq)
        enc_seq = self.encoder(seq, mask, return_embeddings, emb)
        return enc_seq


def get_sinusoid_table(max_len, emb_dim):
    def get_angle(pos, i, emb_dim):
        return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(emb_dim):
            if i % 2 == 0:
                sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
            else:
                sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
    return sinusoid_table
