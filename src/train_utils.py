import math
import torch
import torch.nn.functional as F

def configure_optimizers(model, lr_max=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, fused=False):
    """
    Configures AdamW with weight decay 0.1 on >= 2-D parameters only,
    and 0.0 on 1-D parameters (LayerNorm, biases).
    """
    decay_params = []
    nodecay_params = []
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    extra_args = {}
    if fused and torch.cuda.is_available():
        extra_args["fused"] = True

    optimizer = torch.optim.AdamW(optim_groups, lr=lr_max, betas=betas, eps=eps, **extra_args)
    return optimizer

def get_lr(step, max_steps, warmup_steps=20, lr_max=1e-3, min_lr_ratio=0.1):
    """
    Linear warm-up for warmup_steps, then cosine decay to min_lr_ratio * lr_max.
    """
    min_lr = lr_max * min_lr_ratio
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr_max - min_lr)

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def evaluate_val_loss(model, val_x, val_y, B_eval=16, device="cpu"):
    """
    Validation loss = sum of token cross-entropy / total number of tokens.
    Never average of averages.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    V = model.V

    with torch.no_grad():
        for i in range(0, val_x.size(0), B_eval):
            bx = val_x[i:i+B_eval].to(device)
            by = val_y[i:i+B_eval].to(device)
            logits, _ = model(bx)
            loss_sum = F.cross_entropy(logits.view(-1, V), by.view(-1), reduction="sum")
            total_loss += loss_sum.item()
            total_tokens += by.numel()

    return total_loss / total_tokens
