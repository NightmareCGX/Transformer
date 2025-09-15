import torch
import torch.nn as nn

from .layers import PostNorm, PreNorm
from .utils import build_posenc, causal_mask, padding_mask


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        n_layer: int,
        max_len: int,
        norm: str = "prenorm",
        norm_eps: float = 1e-5,
        attn_bias: bool = True,
        attn_dropout: float = 0.1,
        attn_out_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        ffn_activation: str = "gelu",
        resid_scale: str = "none",
        block_dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        posenc = build_posenc(max_len, d_model)
        self.register_buffer("posenc", posenc, persistent=False)
        self.posenc: torch.Tensor

        if norm.lower() == "prenorm":
            self.layers = nn.ModuleList(
                [
                    PreNorm(
                        d_model,
                        n_head,
                        d_ff,
                        norm_eps,
                        attn_bias,
                        attn_dropout,
                        attn_out_dropout,
                        ffn_dropout,
                        ffn_activation,
                        resid_scale,
                    )
                    for _ in range(n_layer)
                ]
            )
        elif norm.lower() == "postnorm":
            self.layers = nn.ModuleList(
                [
                    PostNorm(
                        d_model, n_head, d_ff, norm_eps, attn_bias, ffn_activation, block_dropout
                    )
                    for _ in range(n_layer)
                ]
            )
        else:
            raise ValueError("unknown norm")

        self.use_final_ln = norm.lower() == "prenorm"
        self.final_ln = nn.LayerNorm(d_model, eps=norm_eps) if self.use_final_ln else nn.Identity()
        self.final_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, key_padding: torch.Tensor | None = None):
        B, L = idx.shape
        x = self.token_embedding(idx) + self.posenc[:, :L, :].to(dtype=idx.dtype)

        causal = causal_mask(L)
        attn = causal if key_padding is None else causal + padding_mask(key_padding)

        for layer in self.layers:
            x = layer(x, attn)

        x = self.final_ln(x)

        return self.final_proj(x)
