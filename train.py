# train.py
from pathlib import Path

import torch
import yaml

from debug.attn_hook_utils import run_and_check
from transformer.device import resolve_device, setup_global_device
from transformer.dtype import setup_global_dtype
from transformer.model import Decoder


def main():
    cfg_path = Path("configs/base.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path.resolve()}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    mdlcfg = cfg.get("model", {}) or {}
    blkcfg = cfg.get("block", {}) or {}

    device = resolve_device(str(cfg.get("device", "auto")))
    setup_global_device(device)
    dtype = str(cfg.get("dtype", "fp32"))
    setup_global_dtype(dtype)
    vocab_size = int(cfg.get("model", {}).get("vocab_size", 32000))
    batch = int(mdlcfg.get("batch_size", 4))
    seq_len = int(mdlcfg.get("seq_len", 256))
    d_model = int(mdlcfg.get("d_model", 512))

    model = (
        Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_head=int(mdlcfg.get("n_head", 8)),
            d_ff=int(mdlcfg.get("d_ff", 2048)),
            n_layer=int(mdlcfg.get("n_layer", 6)),
            max_len=int(mdlcfg.get("max_len", 1024)),
            norm=blkcfg.get("norm", "prenorm"),
            norm_eps=float(blkcfg.get("norm_eps", 1e-5)),
            attn_bias=bool(blkcfg.get("attn", {}).get("bias", True)),
            attn_dropout=float(blkcfg.get("attn", {}).get("dropout", 0.0)),
            attn_out_dropout=float(blkcfg.get("attn", {}).get("out_dropout", 0.0)),
            ffn_activation=blkcfg.get("ffn", {}).get("activation", "gelu"),
            ffn_dropout=float(blkcfg.get("ffn", {}).get("dropout", 0.0)),
            resid_scale=blkcfg.get("prenorm", {}).get("resid_scale", "none"),
            block_dropout=float(blkcfg.get("postnorm", {}).get("block_dropout", 0.0)),
        )
        .to(device)
        .eval()
    )

    context = torch.randint(0, vocab_size, (batch, seq_len))
    key_padding = torch.zeros(batch, seq_len, dtype=torch.bool)
    key_padding[:, -4:] = True

    # with torch.no_grad():
    #    y = model(context, key_padding=key_padding)
    y, attns = run_and_check(model, context, key_padding=key_padding, vocab_size=vocab_size)


if __name__ == "__main__":
    main()
