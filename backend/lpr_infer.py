from __future__ import annotations

import ast
import re
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import torch

try:
    # import only utilities at module import time to avoid heavy deps
    from lprnet.utils import numpy2tensor as _numpy2tensor, decode as _decode
except Exception:
    # last resort: define thin wrappers that error when used
    _numpy2tensor = None
    _decode = None


ROOT = Path(__file__).resolve().parent.parent


def build_args(root: Path | None = None) -> Namespace:
    root = Path(root) if root is not None else ROOT
    cfg = root / "config" / "kor_config.yaml"
    # sensible defaults matching repo config
    args = Namespace()
    args.pretrained = str(root / "backend" / "weights" / "lprnet_kor.pt")
    args.img_size = (100, 50)
    args.dropout_rate = 0.5
    args.weight_decay = 0.00002
    args.lr = 0.001
    args.t_length = 19
    args.chars = []

    if cfg.exists():
        text = cfg.read_text(encoding="utf-8")
        m = re.search(r"pretrained:\s*'([^']+)'", text)
        if m:
            candidate = (root / m.group(1)).resolve()
            if candidate.exists():
                args.pretrained = str(candidate)
            else:
                # try common alternative locations
                alt1 = (root / "backend" / m.group(1)).resolve()
                alt2 = (root / Path(m.group(1)).name).resolve()
                if alt1.exists():
                    args.pretrained = str(alt1)
                elif alt2.exists():
                    args.pretrained = str(alt2)
                else:
                    args.pretrained = str(candidate)

        m = re.search(r"img_size:\s*!!python/tuple\s*\[([^\]]+)\]", text)
        if m:
            try:
                parts = [int(p.strip()) for p in m.group(1).split(",") if p.strip()]
                args.img_size = tuple(parts)
            except Exception:
                pass

        m = re.search(r"chars:\s*\[(.*)\]", text, re.S)
        if m:
            chars_block = "[" + m.group(1) + "]"
            try:
                args.chars = ast.literal_eval(chars_block)
            except Exception:
                args.chars = re.findall(r"'([^']+)'", m.group(1))

    # Ensure chars is not empty: fallback to a simple alnum set if parsing failed
    if not args.chars:
        args.chars = [str(i) for i in range(10)]

    return args


def load_model(args: Namespace) -> Tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # import LPRNet lazily to avoid importing lightning unless model loading is requested
    try:
        from lprnet import LPRNet
    except Exception:
        from lprnet.lprnet import LPRNet

    model = LPRNet(args)
    model.to(device)
    model.eval()

    ckpt_path = Path(args.pretrained)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"LPRNet checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = None
    if isinstance(ckpt, dict):
        # common checkpoint formats
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception:
        # attempt to strip common Lightning/prefix keys
        new_sd = {}
        for k, v in state_dict.items():
            newk = k
            for prefix in ("model.", "LPRNet."):
                if newk.startswith(prefix):
                    newk = newk[len(prefix) :]
            new_sd[newk] = v
        model.load_state_dict(new_sd, strict=False)

    return model, device


def numpy2tensor(img, img_size):
    return _numpy2tensor(img, img_size)


def decode(preds, chars):
    return _decode(preds, chars)
