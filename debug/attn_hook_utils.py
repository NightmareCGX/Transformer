# attn_hook_utils.py
import math

import torch

_ATTENTION_CLASS_NAME = "MultiHeadSelfAttention"
_ATTENTION_CACHE = {}
_HOOK_HANDLES = []


def _is_attention_core_module(m):
    if m is None:
        return False
    return m.__class__.__name__ == _ATTENTION_CLASS_NAME


@torch.no_grad()
def _save_attn_hook(module, inputs, output):
    x = inputs[0]  # [B, L, D]
    try:
        attn_mask = inputs[1]  # [*, *, L, L]
    except IndexError:
        raise IndexError(
            "Insufficient inputs provided. Expected format: [input_tensor, attention_mask], "
            "but only got {} input(s).".format(len(inputs))
        )

    B, L, D = x.shape
    h = module.n_head
    d = module.d_head

    q = module.query(x).view(B, L, h, d).transpose(1, 2)  # [B,h,L,d]
    k = module.key(x).view(B, L, h, d).transpose(1, 2)  # [B,h,L,d]
    # v = module.value(x).view(B, L, h, d).transpose(1, 2)       # [B,h,L,d]

    qf = q.float()
    kf = k.float()

    scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(d)  # [B,h,L,L]
    scores = scores + attn_mask

    attn = torch.softmax(scores, dim=-1)  # [B,h,L,L]

    _ATTENTION_CACHE[id(module)] = attn.detach()


def attach_attention_hooks(model):
    detach_attention_hooks()
    for m in model.modules():
        if _is_attention_core_module(m):
            _HOOK_HANDLES.append(m.register_forward_hook(_save_attn_hook))
    return list(_HOOK_HANDLES)


def detach_attention_hooks():
    global _HOOK_HANDLES, _ATTENTION_CACHE
    for h in _HOOK_HANDLES:
        try:
            h.remove()
        except Exception:
            pass
    _HOOK_HANDLES = []
    _ATTENTION_CACHE = {}


def collect_attn_from(model):
    attns = []
    for m in model.modules():
        if _is_attention_core_module(m):
            attns.append(_ATTENTION_CACHE.get(id(m)))
    return attns


def validate_logits(y: torch.Tensor, batch: int, seq_len: int, vocab_size: int):
    assert y.shape == (
        batch,
        seq_len,
        vocab_size,
    ), f"Logits shape mismatch: got {tuple(y.shape)}, expect {(batch, seq_len, vocab_size)}"
    assert torch.isfinite(y).all().item(), "Found NaN/Inf in logits."


def validate_attentions(attn_list, key_padding: torch.Tensor, atol: float = 1e-5):
    assert any(
        a is not None for a in attn_list
    ), "Unable to capture attentions, Please check the attn_list that hook captured."

    for i, attn in enumerate(attn_list):
        print(f"Check for layer: {i}")
        if attn is None:
            continue

        # 1) sum = 1
        s = attn.sum(dim=-1)  # [B,h,L]
        ones = torch.ones_like(s)
        assert torch.allclose(s, ones, atol=atol), "The sum of the attentions is not equal to 1."

        # 2) key padding
        if key_padding is not None:
            pad = key_padding[:, None, None, :].expand_as(attn)  # [B,h,L,L]
            masked_vals = attn.masked_select(pad)
            if masked_vals.numel() > 0:
                assert (masked_vals <= 1e-6).all(), "The padding mask is not taking effect."

        # 3) causal mask
        B, h, L, _ = attn.shape
        tri = torch.triu(torch.ones(L, L, dtype=torch.bool, device=attn.device), diagonal=1)[
            None, None, :, :
        ].expand(B, h, L, L)
        viol = attn.masked_select(tri)
        if viol.numel() > 0:
            assert (viol <= 1e-6).all(), "The causal mask is not taking effect."


def print_run_summary(device, context, y, vocab_size):
    print("device:", device)
    print("input shape:", tuple(context.shape))
    print("output shape:", tuple(y.shape))
    print("finite:", torch.isfinite(y).all().item())
    print(
        "matches (B, L, vocab_size):",
        tuple(y.shape) == (context.shape[0], context.shape[1], vocab_size),
    )


@torch.no_grad()
def run_and_check(
    model,
    context: torch.Tensor,
    *,
    key_padding: torch.Tensor,
    vocab_size: int,
    print_summary: bool = True,
    atol: float = 1e-5,
):
    attach_attention_hooks(model)
    try:
        y = model(context, key_padding=key_padding)
        attn_list = collect_attn_from(model)

        batch, seq_len = context.shape[:2]
        validate_logits(y, batch, seq_len, vocab_size)
        validate_attentions(attn_list, key_padding, atol=atol)

        if print_summary:
            device = (
                next(model.parameters()).device
                if any(p.requires_grad or p.is_floating_point() for p in model.parameters())
                else context.device
            )
            print_run_summary(device, context, y, vocab_size)
            print("✅ Transformer W1 Passed！")

        return y, attn_list
    finally:
        detach_attention_hooks()
