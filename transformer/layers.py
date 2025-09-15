import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        bias: bool = True,
        attn_drop: float = 0.0,
        out_drop: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        B, L, D = x.shape
        # qkv in shape of B, H, L, D_head
        q = self.query(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        k = self.key(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        v = self.value(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_drop(self.out(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.fully_connected1 = nn.Linear(d_model, d_ff)
        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "geglu":
            self.fc1_g = nn.Linear(d_model, d_ff)
            self.act = nn.GELU()
        elif activation.lower() == "swiglu":
            self.fc1_g = nn.Linear(d_model, d_ff)
            self.act = nn.SiLU()
        else:
            raise ValueError("unknown activation")
        self.fully_connected2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act_name = activation.lower()

    def forward(self, x: torch.Tensor):
        if self.act_name == "gelu":
            return self.drop(self.fully_connected2(self.act(self.fully_connected1(x))))
        elif self.act_name in ["geglu", "swiglu"]:
            a = self.fully_connected1(x)
            g = self.fc1_g(x)
            return self.drop(self.fc2(self.act(g) * a))


class PreNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        norm_eps: float = 1e-5,
        attn_bias: bool = True,
        attn_dropout: float = 0.1,
        attn_out_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        activation: str = "gelu",
        resid_scale: str = "None",
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model, n_head, bias=attn_bias, attn_drop=attn_dropout, out_drop=attn_out_dropout
        )
        self.ffn = FeedForward(d_model, d_ff, dropout=ff_dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

        if resid_scale == "1/sqrt(N)":
            self.alpha_attn = 1.0 / math.sqrt(2.0)
            self.alpha_ffn = 1.0 / math.sqrt(2.0)
            self.learnable = False
        elif resid_scale == "learned":
            self.alpha_attn = nn.Parameter(torch.ones(1))
            self.alpha_ffn = nn.Parameter(torch.ones(1))
            self.learnable = True
        else:
            self.alpha_attn = 1.0
            self.alpha_ffn = 1.0
            self.learnable = False

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        y = x + self.alpha_attn * self.attn(self.norm1(x), attn_mask)
        y = y + self.alpha_ffn * self.ffn(self.norm2(y))
        return y


class PostNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        norm_eps: float = 1e-5,
        attn_bias: bool = True,
        activation: str = "gelu",
        block_dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_head, attn_bias)
        self.ffn = FeedForward(d_model, d_ff, activation=activation)
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.drop = nn.Dropout(block_dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        y = self.norm1(x + self.drop(self.attn(x, attn_mask)))
        y = self.norm2(y + self.drop(self.ffn(y)))
        return y
