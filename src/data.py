import os
import urllib.request
import hashlib
import numpy as np
import torch

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def load_data(data_path="data/input.txt"):
    """
    Loads the Tiny Shakespeare corpus. If not found locally, downloads it.
    Returns (text, chars, stoi, itos, V, train_data, val_data, sha256_hash, byte_len).
    """
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, data_path)
    
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    byte_len = len(text.encode("utf-8"))
    sha256_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    chars = sorted(list(set(text)))
    V = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode entire text into integer tensor
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # 90% train, 10% val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return text, chars, stoi, itos, V, train_data, val_data, sha256_hash, byte_len

def get_batch(step, split, B, T, train_data, val_data):
    """
    Deterministic batch generator using CPU generator seeded by step.
    For train: torch.Generator(device='cpu').manual_seed(1000 + step)
    Offsets: torch.randint(0, len(data) - T - 1, (B,), generator=g)
    """
    data = train_data if split == "train" else val_data
    g = torch.Generator(device="cpu").manual_seed(1000 + step)
    ix = torch.randint(0, len(data) - T - 1, (B,), generator=g)
    x = torch.stack([data[i:i+T] for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    return x, y

def build_fixed_val_set(val_data, T=128, B_eval=16):
    """
    Fixed validation set:
    n_seq = 20 * B_eval = 320 sequences
    offsets = np.linspace(0, len(val) - T - 2, n_seq).astype(int)
    """
    n_seq = 20 * B_eval
    offsets = np.linspace(0, len(val_data) - T - 2, n_seq).astype(int)
    val_x = torch.stack([val_data[o:o+T] for o in offsets])
    val_y = torch.stack([val_data[o+1:o+T+1] for o in offsets])
    return val_x, val_y
